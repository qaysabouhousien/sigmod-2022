package ner;

import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class Train {
    final static String LINE_SEPARATOR = System.lineSeparator();
    private static void prepareTraining() throws IOException {
        var txt = Files.readString(Path.of("laptops_data_IOB.txt"));
        var examples = Arrays.asList(txt.split(LINE_SEPARATOR.repeat(2)));
        Collections.shuffle(examples,new Random(42));
        var trainProp=  (int) (0.7* examples.size());
        var training = examples.subList(0,trainProp);
        var test = examples.subList(trainProp,examples.size());
        System.out.println(training.size());
        System.out.println(test.size());
        System.out.println(training.size()+ test.size() == examples.size());
        var trainTxt = String.join(LINE_SEPARATOR.repeat(2),training);
        Files.writeString(Path.of("laptops_data_train_IOB.txt"),trainTxt);
        var testTxt = String.join(LINE_SEPARATOR.repeat(2),test);
        Files.writeString(Path.of("laptops_data_test_IOB.txt"),testTxt);
    }
    private static void preProcess() throws IOException {
        var txt = Files.readString(Path.of("laptops_data.txt"));
        var examples = txt.split(LINE_SEPARATOR.repeat(2));
        Properties props = new Properties();
        // set the list of annotators to run
//        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,kbp,quote");
        props.setProperty("annotators", "tokenize");
        // set a property for an annotator, in this case the coref annotator is being set to use the neural algorithm
        props.setProperty("coref.algorithm", "neural");
        // build pipeline
        var pipeline = new StanfordCoreNLP(props);
        var IOBFormat = new StringBuilder();
        for(var example : examples){
            var tokens = example.split(LINE_SEPARATOR);
            for(var token : tokens){
                var temp = token.split("\t");
                var t = temp[0];
                var label = temp[1];
                var doc= new CoreDocument(t);
                pipeline.annotate(doc);
                for(var i= 0;i< doc.tokens().size();i++){
                    var prefix = "I_";
                    if (i==0)
                        prefix = "B_";
                    prefix+=label;
                    IOBFormat.append(doc.tokens().get(i));
                    IOBFormat.append("\t");
                    IOBFormat.append(prefix);
                    IOBFormat.append(LINE_SEPARATOR);
                }
            }
            IOBFormat.append(LINE_SEPARATOR);

        }
        Files.writeString(Path.of("laptops_data_IOB.txt"),IOBFormat.toString());

    }
    private static void train() throws Exception {
        var args = new String[]{"-prop", "ner.model.props"};
        CRFClassifier.main(args);
    }
    public static void main(String[] args) throws Exception {
        preProcess();
        prepareTraining();
        train();
    }
}
