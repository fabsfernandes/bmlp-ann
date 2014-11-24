package br.com.ufu.lsi.studies.model;

import java.util.List;

public class NetworkData {
	
	private List<Double> input;
	
	private List<Double> output;
	
	private Double error;
	
	
	public List<Double> getInput() {
		return input;
	}
	
	public void setInput( List<Double> input ) {
		this.input = input;
	}
	
	public List<Double> getOutput() {
		return output;
	}
	
	public void setOutput( List<Double> output ) {
		this.output = output;
	}

	public Double getError() {
		return error;
	}

	public void setError( Double error ) {
		this.error = error;
	}

}
