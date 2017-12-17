package main;

import core.*;

import java.util.ArrayList;
import java.util.HashMap;

public class Bay implements Classifier {
    private String filename;
    private HashMap<String, HashMap<String, Double>> attributes;
    private HashMap<String, Double> instances;
    private int instanceCount;

    public Bay(String filename) {
        this.filename = "src/main/resources/" + filename;
        this.attributes = new HashMap<>();
        this.instances = new HashMap<>();
    }

    public void readData() {
        Evaluator eval = new Evaluator(this, filename);
        eval.evaluateWholeSet();
        eval.evaluateCV();
    }

    /**
     * Trains the classifier.
     *
     * @param train Data set used for training
     */
    @Override
    public void train(Dataset train) {

        // Class values [good, bad]
        train.getDistinctClassValues().getNominalValues().forEach(attribute -> instances.put(attribute,0.0));

        // Attribute name [GamePad, StiffNeck, HoursPlayPerWeek, PlayerSkill]
        train.toList().forEach(attr -> attributes.put(attr.getClassAttribute().nominalValue(), new HashMap<>()));

        instanceCount = train.noInstances();
        for (int i = 0; i < instanceCount; i++) {
            Instance instance = train.getInstance(i);
            String current = instance.getClassAttribute().nominalValue();
            instances.replace(current, instances.get(current) + 1);
            ArrayList<Attribute> attributes = instance.getAttributes();

            for (int i1 = 0; i1 < attributes.size(); i1++) {
                String attr = instance.getAttributeName(i1);
                String nominal = attributes.get(i1).nominalValue();
                System.out.println("Nope");
            }
        }

    }

    /**
     * Classifiers an instance.
     *
     * @param inst The instance
     * @return Predicted class value for the instance
     */
    @Override
    public Result classify(Instance inst) {
        return null;
    }
}
