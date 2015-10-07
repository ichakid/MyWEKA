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


public class C45b extends AbstractClassifier{
	private Attribute attribute;
	private double[] weights;
	private C45b[] branches;
	private List<Double> branchesVal;
	private double classValue;
	private double splitterVal;
	
	/**
	 * for serialization
	 */
	private static final long serialVersionUID = 2404406538125671745L;
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		getCapabilities().testWithFail(data);
		data = new Instances(data);
		makeTree(data);
	}
	
	@Override
	public double classifyInstance(Instance inst) throws Exception {
		if (attribute == null) {
			return classValue;
		}
		//handle numeric attribute here
		if (attribute.isNumeric()) {
			if ((Double) max((inst.value(attribute) - splitterVal), 0.0) != 0.0) {
				return branches[1].classifyInstance(inst);
			} else {
				return branches[0].classifyInstance(inst);
			}
		} else if (inst.isMissing(attribute)){
			return branches[Utils.maxIndex(weights)].classifyInstance(inst);
		} else {
			return branches[branchesVal.indexOf((Double) inst.value(attribute))].classifyInstance(inst);
		}
		
	}

	@Override
	public Capabilities getCapabilities() {
		Capabilities capabilities = new Capabilities(this);
		capabilities.disableAll();
		
		capabilities.enable(Capability.BINARY_CLASS);
		//enable multi class
		
		capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
		capabilities.enable(Capability.NUMERIC_ATTRIBUTES);
		capabilities.enable(Capability.MISSING_VALUES);
		capabilities.enable(Capability.BINARY_ATTRIBUTES);
		return capabilities;
	}
	
	private void makeTree(Instances data) {
		//dataset consists of 1 class
		if (data.numDistinctValues(data.classIndex()) == 1){
			classValue = data.firstInstance().classValue();
			return;
		}
		
		//dataset consists of 1 attribute or doesn't have any instances
		if ((data.numAttributes() < 2) || (data.numInstances() == 0)){
			classValue = mostCommonValue(data, data.classAttribute());
			attribute = null;
			return;
		}
		
		//normal & recursive
		double[] infoGains = new double[data.numAttributes()];
		Enumeration<Attribute> attEnum = data.enumerateAttributes();
		while(attEnum.hasMoreElements()) {
			Attribute att = attEnum.nextElement();
			infoGains[att.index()] = computeInfoGain(data, att);
		}
		attribute = data.attribute(Utils.maxIndex(infoGains));
		if (Utils.eq(infoGains[attribute.index()], 0)){
			attribute = null;
			classValue = mostCommonValue(data, data.classAttribute());
		} else {
			Branch[] dataBranches;
			if (!attribute.isNumeric()) {
				dataBranches = splitData(data, attribute);
			} else {
				dataBranches = splitNumericData(data, attribute);
			}
			branches = new C45b[dataBranches.length];
			branchesVal = new ArrayList<Double>();
			for (int i=0; i<dataBranches.length; i++){
				System.out.println(dataBranches[i].data.numInstances());
				if (dataBranches[i].data.numInstances() > 0){
					if(!attribute.isNumeric())
						branchesVal.add(i, (Double) dataBranches[i].data.firstInstance().value(attribute));
					else {
						//branchesVal=0 for x<=splitterVal, else !=0
						splitterVal = getSplitterVal(dataBranches, attribute);
						branchesVal.add(i, (Double) max((dataBranches[i].data.firstInstance().value(attribute)-splitterVal), 0.0));
					}
				}
				branches[i] = new C45b();
				branches[i].makeTree(dataBranches[i].data);
			}
		}
	}
	
	private double max(double a, double b) {
		if (a > b)
			return a;
		return b;
	}
	
	private double getSplitterVal(Branch[] splittedData, Attribute att) {
		//Just for numeric att; it's guaranteed that splittedData.size() = 2
		splittedData[0].data.sort(att);
		splittedData[1].data.sort(att);
		double low = splittedData[0].data.lastInstance().value(att);
		double high = splittedData[1].data.firstInstance().value(att);
		double retVal = (low + high) * 0.5;
		return retVal;
	}
	
	private double computeInfoGain(Instances data, Attribute att) {
		double gain = computeEntropy(data);
		Branch[] splitData;
		if (!att.isNumeric())
			splitData = splitData(data, att);
		else
			splitData = splitNumericData(data, att);
		for (Branch split : splitData){
			if (split.data.numInstances() > 0){
				gain -= ((double )split.data.numInstances() / (double) data.numInstances())
					* computeEntropy(split.data);
			}
		}
		return gain;
	}
	
	private Branch[] splitNumericData(Instances data, Attribute att) {
		List<Double> attVals = new ArrayList<Double>();
		Branch[] splitData = new Branch[2];
		List<Double> splitterVal = new ArrayList<Double>();
		double[] gains = new double[(data.size()-1)];
		//sort data based on att
		data.sort(att);
		double gain = computeEntropy(data);
		
		//TODO add handler missing Value, in Weka after sorting ascending, instance with missing value is placed in the last row
		//pasti missing value nya bakal ada di splitData[1] dong? is it ok? gimana ngitung splitterVal kalau missing value ada di perbatasan?
		Instances tmpData = new Instances(data);
		tmpData.deleteWithMissing(att);
		
		//do splitting, record splitterVal and IG in each attempt
		for(int i = 0; i < (tmpData.size() - 1); i++) {
			splitData[0] = new Branch(new Instances(tmpData, 0, (i+1)));
			splitData[1] = new Branch(new Instances(tmpData, (i+1), (tmpData.size()-(i+1))));
			splitterVal.add((Double) (tmpData.instance(i).value(att) + tmpData.instance(i+1).value(att)) * 0.5);
			for (Branch split : splitData){
				if (split.data.numInstances() > 0){
					gain -= ((double )split.data.numInstances() / (double) tmpData.numInstances()) //or it should be data.numInstances()
						* computeEntropy(split.data);
					gains[i] = gain;
				}
			}
		}
		//get max gain index, return instances with that splitterVal
		int maxGainIdx = Utils.maxIndex(gains);
		splitData[0] = new Branch(new Instances(tmpData, 0, (maxGainIdx+1)));
		splitData[1] = new Branch(new Instances(tmpData, (maxGainIdx+1), (tmpData.size()-(maxGainIdx+1))));
//		Double spltVal = splitterVal.get(maxGainIdx);
		for (Instance inst: data){
			if (inst.isMissing(att)){
				for (int i = 0; i < 2; i++){ //TODO loop twice
					//TODO if-else condition, compare with spltVal (?)
					splitData[i].addWeight((double) splitData[i].data.size()/(double)tmpData.size());
				}
			}
		}
		
		return splitData;
	}
	
	/*
	 * Same methods with myID3's
	 */
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
	
	private Branch[] splitData(Instances data, Attribute att) {
		List<Double> attVals = new ArrayList<Double>();
		Branch[] splitData = new Branch[att.numValues()];
		for (int i=0; i<att.numValues(); i++) {
			splitData[i] = new Branch(data, data.numInstances());
		}
		Enumeration<Instance> instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = instEnum.nextElement();
			if (!inst.isMissing(att)) {
				Double val = (Double) inst.value(att);
				if (!attVals.contains(val)){
					attVals.add(val);
				}
				splitData[attVals.indexOf(val)].add(inst);
			}
		}
		
		int n = 0;
		for (int i=0; i<att.numValues(); i++) {
			splitData[i].data.compactify();
			n += splitData[i].data.size();
		}
		for (Instance inst: data){
			if (inst.isMissing(att)){
				for (Double val: attVals){
					int idx = attVals.indexOf(val);
					splitData[idx].addWeight((double) splitData[idx].data.size()/(double)n);
				}
			}
		}
		return splitData;
	}
	
	public class Branch{
		public Instances data;
		public double weight;
		
		public Branch(Instances data){
			this.data = new Instances(data);
			weight = 0;
		}
		
		public Branch(Instances data, int n){
			this.data = new Instances(data, n);
			weight = 0;
		}
		
		public void add(Instance inst){
			data.add(inst);
			weight++;
		}
		
		public void addWeight(double w){
			weight += w;
		}
	}
}
