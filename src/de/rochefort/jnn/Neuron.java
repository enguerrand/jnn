package de.rochefort.jnn;

import java.util.Arrays;
import java.util.function.Function;

public class Neuron {
	private double [] inputs;
	private double [] weights;
	private final Function<Double,Double> activationFunction;
	
	public Neuron(Function<Double,Double> activationFunction, int inputCount){
		this.activationFunction = activationFunction;
		weights = new double[inputCount+1];
		for (int i=0; i<inputCount+1; i++){
			weights[i] = NeuralNetwork.RANDOM.nextDouble();
		}
	}
	
	public void setInputs(double[] inputs) {
		if(inputs.length != weights.length - 1){
			throw new IllegalArgumentException("Wrong inputs array size "+inputs.length+". Expected: "+(weights.length-1));
		}
		this.inputs = inputs;
	}

	public double getOutput(){
		double weightedSum=0;
		for (int i=0; i<inputs.length; i++){
			weightedSum += weights[i] * inputs[i];
		}
		weightedSum -= weights[weights.length-1]; // bias
		return activate(weightedSum);
	}
	
	private double activate(double value){
		return this.activationFunction.apply(value);
	}
	
	public double[] getWeights() {
		return weights;
	}
	
	public void setWeights(double[] weights) {
		this.weights = weights;
	}
	
	@Override
	public String toString() {
		return "Neuron [weights=" + Arrays.toString(weights) + "]";
	}
	
}	
	