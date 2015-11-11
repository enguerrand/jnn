package de.rochefort.jnn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
	public static final Random RANDOM = new Random(System.currentTimeMillis());
	private final List<NeuronLayer> neuronLayers = new ArrayList<>();
	NeuralNetwork(List<NeuronLayer> neuronLayers) {
		this.neuronLayers.addAll(neuronLayers);
	}
	
	public double[] getOutput(double[] inputValues){
		double[] inputs = new double[inputValues.length];
		double[] outputs = new double[0];
		System.arraycopy(inputValues, 0, inputs, 0, inputValues.length);
		for(NeuronLayer layer : neuronLayers){
			layer.setInputs(inputs);
			outputs = layer.fire();
			inputs = new double[outputs.length];
			System.arraycopy(outputs, 0, inputs, 0, outputs.length);
		}
		return outputs;
	}
	
	public double[] getWeights(){
		double weights[] = new double[getWeightsLength()];
		int pos = 0;
		for(NeuronLayer nl : this.neuronLayers){
			System.arraycopy(nl.getWeights(), 0, weights, pos, nl.getWeightsLength());
			pos += nl.getWeightsLength();
		}
		return weights;
	}

	public void setWeights(double[] weights){
		if(weights.length != getWeightsLength()){
			throw new IllegalArgumentException("Wrong length of weights array: "+weights.length+" - Expected: "+getWeightsLength());
		}
		int pos = 0;
		for(NeuronLayer nl : this.neuronLayers){
			double layerWeights[] = new double[nl.getWeightsLength()];
			System.arraycopy(weights, pos, layerWeights, 0, nl.getWeightsLength());
			nl.setWeights(layerWeights);
			pos += nl.getWeightsLength();
		}
	}
	
	private int getWeightsLength(){
		int length = 0;
		for(NeuronLayer nl : this.neuronLayers){
			length += nl.getWeightsLength();
		}
		return length;
	}
	
	@Override
	public String toString() {
		return "NeuralNetwork [neuronLayers=" + neuronLayers + "]";
	}
	
	public static void main(String[] args) {
		NeuralNetwork nn = new NetworkBuilder(3, 2)
			.setInputLayerNeuronsCount(4)
			.setDefaultActivationFunctionParameterAllLayers(1)
			.appendHiddenLayer(4)
			.appendHiddenLayer(2)
		.build();
		System.out.println(nn.toString());
		double [] inputs = {3.1,43.2,-2.32};
		System.out.println(Arrays.toString(nn.getOutput(inputs)));
	}
}
