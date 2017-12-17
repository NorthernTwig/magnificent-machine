package main;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.classifiers.functions.MultilayerPerceptron;

public class Neural {
    private String filename;
    private Instances data;
    private Classifier cl;

    public Neural(String filename) {
        this.filename = filename;
    }

    public void readData() {
        try {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(filename);
            data = source.getDataSet();
            data.setClassIndex(8);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    public void train() {
        try {
            cl = new MultilayerPerceptron();
            cl.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void test() {
        try {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(cl, data, 10, new java.util.Random(1));
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
