
import java.util.Random;
import java.util.Scanner;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

public class MyWEKA {
	public Instances data = null;
	public Classifier model = null;
	
	public void loadData(String filename) throws Exception{
		data = DataSource.read(filename);
    }
	
	public void setClassAttribute(int index) {
		data.setClassIndex(index);
	}
	
	public void removeAttribute(String attributeIndex) {
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = attributeIndex;
		Remove remove = new Remove();
		try {
			remove.setOptions(options);
			remove.setInputFormat(data);
			data = Filter.useFilter(data, remove);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void resample(String options) {
		String[] optionsplit = options.split("\\s+");
		Resample resample = new Resample();
		try {
			resample.setOptions(optionsplit);
			resample.setInputFormat(data);
			data = Filter.useFilter(data, resample);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void buildClassifier(String algo) throws Exception{
		if (algo.equals("nb")) {
			model = new NaiveBayes();
		} else if (algo.equals("j48")) {
			model = new J48();
		} else {
			model = new myID3();
		}
		model.buildClassifier(data);
	}
	
	public void testModelGivenDatatest(String datatestFile){
		Instances datatest;
		try {
			datatest = DataSource.read(datatestFile);
			Evaluation eval = new Evaluation(data);
			eval.evaluateModel(model, datatest);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void crossValidation(){
		try {
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(model, data, 10, new Random(1));
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void percentageSplit(double percentage){
		try {
			int trainSize = (int) Math.round(data.numInstances() * percentage / 100); 
			int testSize = data.numInstances() - trainSize; 
			data.randomize(new Random(1));
			Instances train = new Instances(data, 0, trainSize); 
			Instances test = new Instances(data, trainSize, testSize);
			model.buildClassifier(train);
			
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(model, test);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void saveModel(String filename) throws Exception {
		SerializationHelper.write(filename + ".model", model);
    }
	
	public void loadModel(String modelFile) throws Exception{
		model = (Classifier) SerializationHelper.read(modelFile);
	}
	
	public void classify(String unlabeledFile) throws Exception{
		if (model == null){
			System.out.println("model is null");
			return;
		}
        Instances unlabeled = DataSource.read(unlabeledFile);
        unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
        
        Instances labeled = new Instances(unlabeled); // create copy
        
        for (int i = 0; i < unlabeled.numInstances(); i++) {
            double clsLabel = model.classifyInstance(unlabeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        
        DataSink.write("labeled-" + unlabeledFile, labeled);
	}
	
	public static void main(String[] args) throws Exception {
		MyWEKA mw = new MyWEKA();
		Scanner input = new Scanner(System.in);
		System.out.println("Welcome to MyWEKA");
		System.out.println("1. Load Data");
		System.out.println("2. Load Model");
		String cmdString = input.next();
		System.out.println("Input filename: ");
		String filename = input.nextLine();
		switch (cmdString) {
			case "1": mw.loadData("/home/icha/weka-3-7-13/data/weather.nominal.arff");
					  mw.setClassAttribute(4);
					  mw.buildClassifier("id3");
					  mw.crossValidation();
					  break;
			case "2": mw.loadModel(filename);
					  break;
			default:
				break;
		}
	}

}
