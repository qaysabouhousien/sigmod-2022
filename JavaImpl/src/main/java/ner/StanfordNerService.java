package ner;

import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.crf.CRFClassifierWithLOP;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.Triple;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class StanfordNerService implements Serializable {

    private final static String SERIALIZED_CLASSIFIER = "ner.model.ser.gz";

    static AbstractSequenceClassifier<CoreLabel> classifier;

    static {
        try {
            classifier = CRFClassifierWithLOP.getClassifier(SERIALIZED_CLASSIFIER);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
    public static List<String> apply(String text){
        var res = classifier.classifyToCharacterOffsets(text);
        var groupedEntities =  groupIOBEntities(res);
        return toEntities(text, groupedEntities);
    }
    public static Set<String> applyDistinct(String text){
        if (text == null)
            return new HashSet<>();
        var res = classifier.classifyToCharacterOffsets(text);
        var groupedEntities =  groupDistinctIOBEntities(res);
        return toDistinctEntities(text, groupedEntities);
    }
    private static Set<String> toDistinctEntities(String originalText, Set<Triple<String,Integer,Integer>> labels){
        Set<String> entities = new HashSet<>();
        for(var label : labels){
            var e = originalText.substring(label.second,label.third);
            if(e.length() <2)
                continue;
            entities.add(e);
        }
        return entities;
    }
    private static List<String> toEntities(String originalText, List<Triple<String,Integer,Integer>> labels){
        List<String> entities = new ArrayList<>();
        for(var label : labels){
            var e = originalText.substring(label.second,label.third);
            if(e.length() <2)
                continue;
            entities.add(e);
        }
        return entities;
    }
    private static Set<Triple<String,Integer,Integer>> groupDistinctIOBEntities(List<Triple<String,Integer,Integer>> res){
        Set<Triple<String,Integer,Integer>> entities = new HashSet<>();
        String currentEntity = null;
        int currentStart = 0;
        int currentEnd = 0;
        for (var e : res) {
            var label = e.first;
            var entity = label.substring(2);
            if (!entity.equals(currentEntity)) {
                if (currentEntity != null)
                    entities.add(Triple.makeTriple(currentEntity, currentStart, currentEnd));
                currentEntity = entity;
                currentStart = e.second;
            }
            currentEnd = e.third;
        }
        entities.add(Triple.makeTriple(currentEntity, currentStart, currentEnd));

        return entities;

    }

    private static List<Triple<String,Integer,Integer>> groupIOBEntities(List<Triple<String,Integer,Integer>> res){
        List<Triple<String,Integer,Integer>> entities = new ArrayList<>();
        String currentEntity = null;
        int currentStart = 0;
        int currentEnd = 0;
        for (var e : res) {
            var label = e.first;
            var entity = label.substring(2);
            if (!entity.equals(currentEntity)) {
                if (currentEntity != null)
                    entities.add(Triple.makeTriple(currentEntity, currentStart, currentEnd));
                currentEntity = entity;
                currentStart = e.second;
            }
            currentEnd = e.third;
        }
        entities.add(Triple.makeTriple(currentEntity, currentStart, currentEnd));

        return entities;

    }

}
