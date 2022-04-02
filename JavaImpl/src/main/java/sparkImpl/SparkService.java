package sparkImpl;
import ner.StanfordNerService;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.fpm.FPGrowth;
import org.apache.spark.ml.fpm.FPGrowthModel;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.udf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class SparkService {
    private static final SparkSession spark = SparkSession.builder()
            .appName("Simple Application")
            .config("spark.master","local")
            .config("spark.ui.enabled","false")
            .config("spark.executor.processTreeMetrics.enabled","false")
            .getOrCreate();
    static {
        registerNER_UDF();
    }
    public static void main(String[] args) throws IOException {
        runOnDataset("X1.csv","title","Y1.csv");
        runOnDataset("X2.csv","name","Y2.csv");
    }
    public static Set<Set<String>> collapseSet(Set<Set<String>> set){
        return set.stream().filter(s1 -> set.stream().noneMatch(s2 -> s1.containsAll(s2) && !s1.equals(s2))).collect(Collectors.toSet());
    }
    static Dataset<Row> readFile(String path){
        var df = spark.read().option("header","true").option("inferSchema","true").csv(path);
        df = df.withColumn("id", df.col("id").cast("int"));
//        for(var i = 0;i <1;i++){
//           df = df.union(spark.read().option("header","true").option("inferSchema","true").csv(path));
//        }
        System.out.println(df.count());
        return df;
    }
    private static void registerNER_UDF(){
        var res = udf((UDF1<String, Set<String>>) StanfordNerService::applyDistinct, DataTypes.createArrayType(DataTypes.StringType));
        spark.sqlContext().udf().register("ner_func",res);
    }
    private static Dataset<Row> applyNER_UDF(Dataset<Row> df,String columnName){
        StopWatch w = new StopWatch();
        w.start();
        df=  df.withColumn("items", functions.call_udf("ner_func",df.col(columnName))).cache();
        w.stop();
        System.out.println(w);
        System.out.println("NER TIME : "+w);

        return df;
    }
    private static Dataset<Row> fpGrowth(Dataset<Row> df, String itemsColumn, double minSupport){
        StopWatch w = new StopWatch();
        w.start();
        FPGrowthModel model = new FPGrowth()
                .setItemsCol(itemsColumn)
                .setMinSupport(minSupport)
                .setMinConfidence(1)
                .fit(df);
        var fi = model.freqItemsets();
        w.stop();
        System.out.println("FP_GROWTH TIME : "+w);
        return fi;
    }
    private static Set<Tuple<Integer,Integer>> doBlocking(Dataset<Row> df, Dataset<Row> freqItems){
        var frequentSets =  collapseSet(freqItems.select("items")
                .collectAsList()
                .parallelStream()
                .map(l -> l.getList(0))
                .map(l -> {
                    Set<String> outList = new HashSet<>();
                    for(var val : l)
                        outList.add(val.toString());
                    return outList;
                })
                .collect(Collectors.toSet()));
        var da = df.select("id","items").map((MapFunction<Row, FrequentItems>) row ->{
            var o =row.get(0);
            if(o == null)
                return new FrequentItems(0,new LinkedList<>());
            var id = row.getInt(0);
            var s = Set.copyOf(row.getList(1));
            var sets = frequentSets.parallelStream().filter(s::containsAll).map(List::copyOf).collect(Collectors.toList());
            return new FrequentItems(id,sets);
        },Encoders.bean(FrequentItems.class)).filter((FilterFunction<FrequentItems>)  i -> i.sets.size() > 0).collectAsList();
        Map<List<String>, List<Integer>> blocks = new HashMap<>();
        for(var fq : da){
            for(var set : fq.sets){
                var arr = blocks.getOrDefault(set,new ArrayList<>());
                arr.add(fq.id);
                blocks.put(set,arr);
            }
        }
        blocks.values().forEach(b -> System.out.println(b.size()));
        Set<Tuple<Integer,Integer>> pairs= new HashSet<>();
        for(var r : blocks.entrySet()){
            var values = r.getValue();
            values.sort(Comparator.naturalOrder());
            for(var i =0;i< values.size();i++){
                for(var j =i+1;j< values.size();j++) {
                    pairs.add(Tuple.create(values.get(i),values.get(j)));
                }
            }
        }
        return pairs;
    }
    public static void runOnDataset(String fileName, String columnName, String evaluationFileName) throws IOException {
        StopWatch w = new StopWatch();
        w.start();
        var dataset = readFile(fileName);
        dataset = applyNER_UDF(dataset,columnName);
        dataset.show(false);
//        var freqItems  = fpGrowth(dataset,"items",0.0005).cache();
//        Set<Tuple<Integer,Integer>> pairs= doBlocking(dataset,freqItems);
//        System.out.println(pairs.size());
//        evaluate(evaluationFileName, pairs);
        w.stop();
        System.out.println("total time "+ w);
    }

    private static void evaluate(String evaluationFileName, Set<Tuple<Integer,Integer>> pairs) throws IOException {
        var y1 = Files.lines(Path.of(evaluationFileName));
        var trueTuples = y1.skip(1).map(l ->{
            var t = l.split(",");
            return Tuple.create(Integer.parseInt(t[0]),Integer.parseInt(t[1]));
        }).collect(Collectors.toSet());
        var r= computeRecall(pairs,trueTuples);
        System.out.println(r);
    }
    private static double computeRecall(Set<Tuple<Integer,Integer>> predicted, Set<Tuple<Integer,Integer>> truth){
        int  c = 0;
        for(var t : truth){
            if(predicted.contains(t)){
                c++;
            }
        }
        return c/(double)truth.size();
    }

}