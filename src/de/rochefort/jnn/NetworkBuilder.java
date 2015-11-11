package de.rochefort.jnn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;

import de.rochefort.jnn.NeuronLayer.NeuronLayerType;

public class NetworkBuilder {
	private final int inputsCount;
	private final int outputsCount;
	private int inputLayerNeuronCount;
	private HashMap<NeuronLayerType, Function<Double, Double>> activationFunctions = new HashMap<>();
	private List<Integer> layerNeuronCounts = new ArrayList<>();
	public NetworkBuilder(int inputsCount, int outputsCount) {
		setDefaultActivationFunctionParameterAllLayers(1);
		this.inputsCount = inputsCount;
		this.inputLayerNeuronCount = inputsCount;
		this.outputsCount = outputsCount;
	}
	
	public NetworkBuilder setDefaultActivationFunctionParameterAllLayers(double p){
		return setActivationFunctionAllLayers(getDefaultActivationFunction(p));
	}
	
	public NetworkBuilder setDefaultActivationFunctionParameter(double p, NeuronLayerType neuronLayerType){
		activationFunctions.put(neuronLayerType, getDefaultActivationFunction(p));
		return this;
	}
	
	public NetworkBuilder setActivationFunctionAllLayers(Function<Double,Double> activationFunction){
		for(NeuronLayerType t : NeuronLayerType.values()){
			setActivationFunction(t, activationFunction);
		}
		return this;
	}
	
	public NetworkBuilder setActivationFunction(NeuronLayerType neuronLayerType, Function<Double,Double> activationFunction){
		activationFunctions.put(neuronLayerType, activationFunction);
		return this;
	}
	
	public NetworkBuilder appendHiddenLayer(int neuronCount){
		this.layerNeuronCounts.add(neuronCount);
		return this;
	}
	
	public NetworkBuilder setInputLayerNeuronsCount(int neuronsCount) {
		this.inputLayerNeuronCount = neuronsCount;
		return this;
	}
	
	private static Function<Double, Double> getDefaultActivationFunction(double activationParam){
		return input -> 1.0 / (1 + Math.pow(Math.E, (- input) / activationParam));
	}

	public NeuralNetwork build(){
		List<NeuronLayer> layers = new ArrayList<>();
		
		NeuronLayer inputLayer = new NeuronLayer(activationFunctions.get(NeuronLayerType.INPUT), inputLayerNeuronCount, inputsCount);
		layers.add(inputLayer);
		
		int inputCount = inputLayer.getNeuronCount();
		for(Integer neuronCount : layerNeuronCounts){
			NeuronLayer hiddenLayer = new NeuronLayer(activationFunctions.get(NeuronLayerType.HIDDEN), neuronCount, inputCount);
			layers.add(hiddenLayer);
			inputCount = neuronCount;
		}
		
		NeuronLayer outputLayer = new NeuronLayer(activationFunctions.get(NeuronLayerType.OUTPUT), outputsCount, inputCount);
		layers.add(outputLayer);
		
		return new NeuralNetwork(layers);
	}
}
