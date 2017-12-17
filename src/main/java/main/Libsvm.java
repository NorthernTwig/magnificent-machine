package main;

import core.*;
import libsvm.*;

import java.io.File;

public class Libsvm implements Classifier {
    private String filename;
    private Dataset data;
    private svm_problem prob;
    private svm_model model;

    public Libsvm(String filename) {
        this.filename = "src/main/resources/" + filename;
    }

    public void readData() throws Exception {
        File file = new File(filename);
        String path = file.getAbsolutePath();
        Evaluator evaluator = new Evaluator(this, path);
        evaluator.evaluateWholeSet();
    }

    public void train(Dataset train) {
        data = train;
        int n = data.noInstances();
        svm_problem prob = new svm_problem();
        prob.y = new double[n];
        prob.l = n;
        prob.x = new svm_node[n][data.noAttributes() - 1];
        for (int i = 0; i < data.noInstances(); i++) {
            Instance inst = data.getInstance(i);

            double[] vals = inst.getAttributeArrayNumerical();
            prob.x[i] = new svm_node[data.noAttributes() - 1];

            for (int a = 0; a < data.noAttributes() - 1; a++) {
                svm_node node = new svm_node();
                node.index = a;
                node.value = vals[a];
                prob.x[i][a] = node;
            }

            prob.y[i] = inst.getClassAttribute().numericalValue();
        }

        svm_parameter param = new svm_parameter();
        param.probability = 1;
        param.gamma = 0.5;
        param.nu = 0.5;
        param.C = 100;
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.RBF;
        param.cache_size = 20000;
        param.eps = 0.001;

        model = svm.svm_train(prob, param);
    }


    /**
     * Classifiers an instance.
     *
     * @param inst The instance
     * @return Predicted class value for the instance
     */
    public Result classify(Instance inst) {
        double[] vals = inst.getAttributeArrayNumerical();
        int no_classes = data.noClassValues();

        svm_node[] nodes = new svm_node[vals.length];
        for (int a = 0; a < vals.length; a++) {
            svm_node node = new svm_node();
            node.index = a;
            node.value = vals[a];
            nodes[a] = node;
        }

        int[] labels = new int[no_classes];
        svm.svm_get_labels(model, labels);
        double[] prob_estimates = new double[no_classes];
        double cVal = svm.svm_predict_probability(model, nodes, prob_estimates);
        return new Result(cVal);
    }
}
