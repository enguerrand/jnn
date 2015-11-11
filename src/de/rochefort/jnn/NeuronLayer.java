package de.rochefort.jnn;

import java.util.Arrays;
import java.util.function.Function;

public class NeuronLayer {
	public enum NeuronLayerType {
		INPUT, HIDDEN, OUTPUT
	}
	private final Neuron[] neurons;
	public NeuronLayer(Function<Double,Double> activationFunction, int neuronCount, int inputCount) {
		this.neurons = new Neuron[neuronCount];
		for(int i = 0; i<neuronCount; i++){
			this.neurons[i] = new Neuron(activationFunction, inputCount);
		}
	}
	
	public void setInputs(double[] inputs) {
		for(int i = 0; i<this.neurons.length; i++){
			this.neurons[i].setInputs(inputs);
		}
	}
	
	public double[] fire(){
		double[] outputs = new double[this.neurons.length]; 
		for(int i = 0; i < this.neurons.length; i++){
			outputs[i] = this.neurons[i].getOutput();
		}
		return outputs;
	}
	
	public int getNeuronCount(){
		return this.neurons.length;
	}
	
	public double[] getWeights() {
		double weights[] = new double[getWeightsLength()];
		for (int i=0; i<this.neurons.length; i++){
			for (int j=0; j < this.neurons[i].getWeights().length; j++){
				weights[i*this.neurons[i].getWeights().length + j ] = this.neurons[i].getWeights()[j];
			}
		}
		return weights;
	}
	
	public int getWeightsLength(){
		return this.neurons.length * this.neurons[0].getWeights().length;
	}
	
	public void setWeights(double[] weights) {
		for (int i=0; i<this.neurons.length; i++){
			double[] newWeights = new double[this.neurons[i].getWeights().length];
			for (int j=0; j < newWeights.length; j++)
			{
				newWeights[j] = weights[i*this.neurons[i].getWeights().length + j ];
			}
			this.neurons[i].setWeights(newWeights);
		}
	}
	
	@Override
	public String toString() {
		return "NeuronLayer [neurons=" + Arrays.toString(neurons) + "]";
	}
}
