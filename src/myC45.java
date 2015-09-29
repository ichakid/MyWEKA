import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.j48.BinC45Split;
import weka.classifiers.trees.j48.Distribution;
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
	private double[] numericSplitter;
	
	private static final long serialVersionUID = 5L;
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		getCapabilities().testWithFail(data);
//		handleMissingValue(data);
		
	    data = new Instances(data);
	    data.deleteWithMissingClass();
		makeTree(data);
	}
	
	@Override
	public double classifyInstance(Instance inst) throws Exception {
		// TODO Auto-generated method stub
		if (attribute == null) {
			return classValue;
		}
		return branches[branchesVal.indexOf((Double) inst.value(attribute))].classifyInstance(inst);
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
//		System.out.println("numclasses " + data.numClasses());
		int nc = data.numClasses();
		int[] classCount = new int[nc];
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
				double temp = (double)classCount[i]/(double)data.numInstances();
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
		for (int i=0; i<att.numValues(); i++) {
			splitData[i].compactify();
		}
		return splitData;
	}
	
	private double[] handleNumericAttr(Instances data, Attribute att) {
		int dataSize = data.size();		
		double[] ret = new double[2]; //untuk nyimpan nilai batas dan infogain
		double[] splitValue = new double[dataSize - 1];
		double[] infoGains = new double[dataSize - 1];
		// sort numericAttr + classAttr
		data.sort(att.index());
		// cari spot perubahan kelas
		// split jadi 2 (A <= h & A > h)
		// itung info gainnya
		for (int i = 1; i < dataSize; i++) {
			int realIdx = i - 1;
			splitValue[realIdx] = (data.get(realIdx).value(att) + data.get(i).value(att)) * 0.5;
			Instances[] tmp = binarySplit(data, i);
			double gain = computeEntropy(data);
			System.out.println(tmp[0].size());
			for (Instances split : tmp){
					gain -= ((double)split.numInstances() / (double) data.numInstances())
						* computeEntropy(split);
			}
			infoGains[realIdx] = gain;
		}
		// return yang info gainnya paling tinggi
		int idxOfMax = getMaxValue(infoGains);
		ret[0] = splitValue[idxOfMax];
		ret[1] = infoGains[idxOfMax];
//		System.out.println("split val " + ret[0] + " ig " + ret[1]);
		return ret;
	}
	
	public int getMaxValue(double[] d) {
		int idx = 0;
		for (int i = 1; i < d.length; i++) {
			if (d[i] > d[idx]) {
				idx = i;
			}
		}
		return idx;
	}
	
	private Instances[] binarySplit(Instances data, int nFirstInstances) {
		int dataSize = data.numInstances();
		Instances[] splitData = new Instances[2];
		for (int i=0; i<2; i++) {
			splitData[i] = new Instances(data, data.numInstances());
		}
		for (int j = 0; j < nFirstInstances; j++) {
			Instance tmp = data.instance(j);	
			splitData[0].add(tmp);
//			System.out.println("nilai atribut numerik " + tmp.value(1));
//			System.out.println("nilai atribut numerik2 " + splitData[0].get(j).value(1));
		}
		for (int i = nFirstInstances; i < dataSize; i++) {
			Instance tmp2 = data.instance(i);
			splitData[1].add(tmp2);
		}
		for (int i=0; i<2; i++) {
			splitData[i].compactify();
		}
		return splitData;
	}
	
	private void handleMissingValue(Instances data, Attribute att) {
		int distinct = att.numValues();
		List<Double> pVal = new ArrayList<Double>();
		
		for (Instance i : data) {
			if (!i.isMissing(att)) {
				if (!pVal.contains((Double) i.value(att))) {
					pVal.add((Double) i.value(att));
				}
			}
		}
	}
	
	private void treePruning(myC45 tree) {
		
	}
	
	private void makeTree(Instances data) {
		if (data.numDistinctValues(data.classIndex()) == 1){
			classValue = data.firstInstance().classValue();
			return;
		}
		
		if ((data.numAttributes() < 2) || (data.numInstances() == 0)){	//leaf node
			classValue = mostCommonValue(data, data.classAttribute());
			attribute = null;
			return;
		}
		
		double[] infoGains = new double[data.numAttributes()];
		Enumeration<Attribute> attEnum = data.enumerateAttributes();
		while (attEnum.hasMoreElements()) {
			Attribute att = (Attribute) attEnum.nextElement();
			if (!att.isNumeric()) {
                             
			} else {
//				System.out.println("nilai infogain numerik " + handleNumericAttr(data, att)[1]);
				numericSplitter = handleNumericAttr(data, att);
				infoGains[att.index()] = numericSplitter[1];
			}
		}
		attribute = data.attribute(Utils.maxIndex(infoGains));
		
		if (Utils.eq(infoGains[attribute.index()], 0)){
			  attribute = null;
			  classValue = mostCommonValue(data, data.classAttribute());
		} else {
			Instances[] dataBranches;
 			if (!attribute.isNumeric()) {
				dataBranches = splitData(data, attribute);
			} else {
				double splitVal = numericSplitter[0];
				System.out.println(splitVal);
				dataBranches = splitWithValue(data, splitVal, attribute);
			}
			branches = new myC45[dataBranches.length];
			branchesVal = new ArrayList<Double>();
			for (int i=0; i<dataBranches.length; i++){
				branchesVal.add(i, (Double) dataBranches[i].firstInstance().value(attribute));
				branches[i] = new myC45();
				branches[i].makeTree(dataBranches[i]);
			}
		}
	}
	
	private Instances[] splitWithValue(Instances data, double value, Attribute att) {
		int dataSize = data.numInstances();
		Instances[] splitData = new Instances[2];
		for (int i=0; i<2; i++) {
			splitData[i] = new Instances(data, data.numInstances());
		}
		for (int j = 0; j < dataSize; j++) {
			Instance tmp = data.get(j);
			if (tmp.value(att) <= value) {
				splitData[0].add(tmp);
			} else {
				splitData[1].add(tmp);
			}
		}
		for (int i=0; i<2; i++) {
			splitData[i].compactify();
		}
		return splitData;
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
		return classes.get(Utils.maxIndex(classCount));
	}
}
