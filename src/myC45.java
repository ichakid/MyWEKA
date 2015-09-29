import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.Capabilities.Capability;


public class myC45 extends AbstractClassifier{
	private Attribute attribute;
	private myC45[] branches;
	private List<Double> branchesVal;
	private double classValue;
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public double classifyInstance(Instance inst) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		Capabilities capabilities = new Capabilities(this);
		capabilities.disableAll();
		capabilities.enable(Capability.BINARY_CLASS);
		capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
		capabilities.enable(Capability.NUMERIC_ATTRIBUTES);
		return capabilities;
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
				double temp = classCount[i]/data.numInstances();
				entropy += -temp * Utils.log2(temp);
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
//		for (int i=0; i<att.numValues(); i++) {
//			splitData[i].compactify();
//		}
		return splitData;
	}
	
	private void handleNumericAttr(Instance data) {
		
	}
	
	private void handleMissingValue(Instance data) {
		
	}
	
	private void treePruning(myC45 tree) {
		
	}
}
