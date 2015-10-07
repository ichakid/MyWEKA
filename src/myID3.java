import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;


public class myID3 extends AbstractClassifier{
	private Attribute attribute;
	private myID3[] branches;
	private List<Double> branchesVal;
	private double classValue;

	/**
	 * for serialization
	 */
	private static final long serialVersionUID = 2404406538125671745L;

	@Override
	public void buildClassifier(Instances data) throws Exception {
		getCapabilities().testWithFail(data);
		handleMissingValue(data);
		
	    data = new Instances(data);
	    data.deleteWithMissingClass();
		makeTree(data);		
	}

	private void handleMissingValue(Instances data) {
		double[] commonValues = new double[data.numAttributes()];
		for (int i=0; i<data.numAttributes(); i++){
			commonValues[i] = mostCommonValue(data, data.attribute(i));
		}
		for (Instance inst: data) {
			if (inst.hasMissingValue()){
				for (int i=0; i<data.numAttributes(); i++){
					if (inst.isMissing(i)){
						inst.setValue(i, commonValues[i]);
					}
				}
			}
		}
	}

	private void makeTree(Instances data) {
		if (data.numDistinctValues(data.classIndex()) == 1){
			classValue = data.firstInstance().classValue();
			return;
		}
		
		if ((data.numAttributes() < 2) || (data.numInstances() == 0)){
			classValue = mostCommonValue(data, data.classAttribute());
			attribute = null;
			return;
		}
		
		double[] infoGains = new double[data.numAttributes()];
		Enumeration<Attribute> attEnum = data.enumerateAttributes();
		while (attEnum.hasMoreElements()) {
			Attribute att = (Attribute) attEnum.nextElement();
			infoGains[att.index()] = computeInfoGain(data, att);
		}
		attribute = data.attribute(Utils.maxIndex(infoGains));
		
		if (Utils.eq(infoGains[attribute.index()], 0)){
			  attribute = null;
			  classValue = mostCommonValue(data, data.classAttribute());
		} else {
			Instances[] dataBranches = splitData(data, attribute);
			branches = new myID3[dataBranches.length];
			branchesVal = new ArrayList<Double>();
			for (int i=0; i<dataBranches.length; i++){
				System.out.println(dataBranches[i].numInstances());
				if (dataBranches[i].numInstances() > 0){
					branchesVal.add(i, (Double) dataBranches[i].firstInstance().value(attribute));
				}
				branches[i] = new myID3();
				branches[i].makeTree(dataBranches[i]);
			}
		}
	}

	private double mostCommonValue(Instances data, Attribute att) {
		int[] classCount = new int[att.numValues()];
		List<Double> classes = new ArrayList<Double>();
		Enumeration<Instance> instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance instance = instEnum.nextElement();
			if (!instance.isMissing(att)){
				Double classVal = (Double) instance.value(att);
				if (!classes.contains(classVal)){
					classes.add(classVal);
				}
				classCount[classes.indexOf(classVal)]++;
			}
		}
		if (classes.size() > 0){
			return classes.get(Utils.maxIndex(classCount));
		} else {
			return Double.NaN;
		}
	}

	private double computeInfoGain(Instances data, Attribute att) {
		double gain = computeEntropy(data);
		Instances[] splitData = splitData(data, att);
		for (Instances split : splitData){
			if (split.numInstances() > 0){
				gain -= ((double )split.numInstances() / (double) data.numInstances())
					* computeEntropy(split);
			}
		}
		return gain;
	}

	private double computeEntropy(Instances data) {
		double entropy = 0;
		int numClasses = data.numClasses();
		int[] classCount = new int[numClasses];
		List<Double> classes = new ArrayList<Double>();
		Enumeration<Instance> instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance instance = instEnum.nextElement();
			Double classVal = (Double) instance.classValue();
			if (!classes.contains(classVal)){
				classes.add(classVal);
			}
			classCount[classes.indexOf(classVal)]++;
		}
		for (Double classVal: classes){
			int i = classes.indexOf(classVal);
			if (classCount[i] > 0) {
				double temp = (double) classCount[i] / (double) data.numInstances();
				entropy += (-1) * temp * Utils.log2(temp);
			}
		}
		return entropy;
	}

	private Instances[] splitData(Instances data, Attribute att) {
		List<Double> attVals = new ArrayList<Double>();
		Instances[] splitData = new Instances[att.numValues()];
		for (int i=0; i<att.numValues(); i++) {
			splitData[i] = new Instances(data, data.numInstances());
		}
		Enumeration<Instance> instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = instEnum.nextElement();
			Double val = (Double) inst.value(att);
			if (!attVals.contains(val)){
				attVals.add(val);
			}
			splitData[attVals.indexOf(val)].add(inst);
		}
		for (int i=0; i<att.numValues(); i++) {
			splitData[i].compactify();
		}
		return splitData;
	}

	@Override
	public double classifyInstance(Instance inst) throws Exception {
		if (attribute == null) {
			return classValue;
		}
		return branches[branchesVal.indexOf((Double) inst.value(attribute))].classifyInstance(inst);
	}

	@Override
	public Capabilities getCapabilities() {
		Capabilities capabilities = new Capabilities(this);
		capabilities.disableAll();
		
		capabilities.enable(Capability.BINARY_CLASS);
		//enable multi class
		
		capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
		return capabilities;
	}
}
