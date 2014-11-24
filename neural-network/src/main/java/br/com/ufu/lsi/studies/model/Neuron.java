package br.com.ufu.lsi.studies.model;

import java.util.List;

public class Neuron {
	
	private Double sumValue;
	
	private Double activationValue;
	
	private List<Double> weights;
	
	private List<Double> previousWeightsVariations;
	
	private Double bias;
	
	private Double previousBiasVariation;
	
	private Double delta;
	

	public List<Double> getWeights() {
		return weights;
	}

	public void setWeights( List<Double> weights ) {
		this.weights = weights;
	}


	public Double getBias() {
		return bias;
	}

	public void setBias( Double bias ) {
		this.bias = bias;
	}

	public Double getSumValue() {
		return sumValue;
	}

	public void setSumValue( Double sumValue ) {
		this.sumValue = sumValue;
	}

	public Double getActivationValue() {
		return activationValue;
	}

	public void setActivationValue( Double activationValue ) {
		this.activationValue = activationValue;
	}

	public Double getDelta() {
		return delta;
	}

	public void setDelta( Double delta ) {
		this.delta = delta;
	}

	public List<Double> getPreviousWeightsVariations() {
		return previousWeightsVariations;
	}

	public void setPreviousWeightsVariations( List<Double> previousWeightsVariations ) {
		this.previousWeightsVariations = previousWeightsVariations;
	}

	public Double getPreviousBiasVariation() {
		return previousBiasVariation;
	}

	public void setPreviousBiasVariation( Double previousBiasVariation ) {
		this.previousBiasVariation = previousBiasVariation;
	}

}
