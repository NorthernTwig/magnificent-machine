package main;

public class Main {
    public static void main(String[] args) throws Exception {
        System.out.println("BAYES");
        Bayes wiki = new Bayes("wikipedia_70.arff");
        wiki.readData();
        wiki.train();
        wiki.test();

        System.out.println("\nTREE");
        Trees fifa = new Trees("FIFA_skill.arff");
        fifa.readData();
        fifa.train();
        fifa.test();

        System.out.println("\nLIBSVM");
        Libsvm matcherTraining = new Libsvm("matchmaker_fixed.arff");
        matcherTraining.readData();

        System.out.println("\nNEURAL");
        Neural neural = new Neural("matchmaker_fixed.arff");
        neural.readData();
        neural.train();
        neural.test();

//        Bay bay = new Bay("FIFA_skill_nominal.arff");
//        bay.readData();
    }
}
