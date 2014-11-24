package br.com.ufu.lsi.studies.model;

import java.util.ArrayList;
import java.util.List;

public class Layer {

	private List<Neuron> neurons;

	public Layer( int neurons ) {

		this.neurons = new ArrayList<Neuron>();
		for ( int i = 0; i < neurons; i++ )
			this.neurons.add( null );
	}

	public void setNeuronsActivationValues( List<Double> values ) {

		for ( int i = 0; i < neurons.size(); i++ ) {
			Neuron neuron = neurons.get( i );
			neuron.setActivationValue( values.get( i ) );
			neurons.set( i, neuron );
		}
	}

	public List<Neuron> getNeurons() {
		return neurons;
	}

	public void setNeurons( List<Neuron> neurons ) {
		this.neurons = neurons;
	}

}
