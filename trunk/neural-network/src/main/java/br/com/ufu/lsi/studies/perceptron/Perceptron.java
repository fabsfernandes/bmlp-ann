package br.com.ufu.lsi.studies.perceptron;

import java.util.ArrayList;
import java.util.List;

import br.com.ufu.lsi.studies.model.NetworkData;


/**
 * Simple Perceptron - input layer and output layer
 * Training with iteractions
 * 
 * Training rate = 1
 * Initial weights = 0
 * Iteractions = 100
 * Activation function = step function
 * Error = expected - obtained
 * 
 * @author fabiola
 *
 */

public class Perceptron {

	List<Double> weights;
	Double trainingRate;
	Double bias;
	int inputNeurons;
	int outputNeurons;
	int MAX_ITERATIONS = 100;

	
	public Perceptron( int inputNeurons, int outputNeurons ) {

		this.inputNeurons = inputNeurons;
		this.outputNeurons = outputNeurons;
	}

	
	public static void main( String... args ) {

		Perceptron p = new Perceptron( 2, 1 );

		List<NetworkData> trainingSet = p.setupTrainingSet();

		p.trainNetwork( trainingSet );
	}

	
	public void trainNetwork( List<NetworkData> trainingSet ) {
		
		initializeParameters();
		
		int iterations = -1;
		boolean zeroError = true;
		
		do {
			iterations++;
			zeroError = true;
			
			System.out.println( "\n## Iteration " + iterations );
			
			for( NetworkData data : trainingSet ) {
				
				List<Double> input = data.getInput();				
				List<Double> expectedOutput = data.getOutput();
				
				Double joinSum = joinSum( weights, input );
				List<Double> gotOutput = activationFunction( joinSum );

				Double currentError = calculateError( expectedOutput, gotOutput );
				
				if( currentError != 0.0 )
					zeroError = false;
				
				for( Double weight : weights ) {
					System.out.print( weight + "\t" );
				}
				System.out.println( bias + " ===> error = " + currentError );
				
				updateWeights( currentError, input );
			}
		} while( iterations < MAX_ITERATIONS && !zeroError );
	}
	
	/**
	 * Initialize all with 0
	 * Training rate = 1
	 */
	public void initializeParameters() {

		trainingRate = 1.0;
		bias = 0.0;

		weights = new ArrayList<Double>();
		for ( int i = 0; i < inputNeurons; i++ ) {
			weights.add( new Double( 0.0 ) );
		}
	}

	public void updateWeights( Double currentError, List<Double> input ) {

		for ( int i = 0; i < input.size(); i++ ) {
			Double newWeight = weights.get( i )
					+ (trainingRate * currentError * input.get( i ));
			weights.set( i, newWeight );
		}
		bias = ( bias + (trainingRate * currentError * 1 ) );
	}

	public Double calculateError( List<Double> di, List<Double> yi ) {

		return di.get( 0 ) - yi.get( 0 );
	}

	public Double joinSum( List<Double> weights, List<Double> input ) {
		
		Double joinSum = 0.0;
		
		for( int i = 0; i < weights.size(); i++ ) {
			joinSum += ( weights.get( i )*input.get( i ) );
		}
		joinSum += bias;
		
		return joinSum;
	}

	/**
	 * Step function 
	 * 
	 * @param joinSum
	 * @return
	 */
	public List<Double> activationFunction( Double joinSum ) {
		
		List<Double> output = new ArrayList<Double>();
		
		if( joinSum >= 0.0 )
			output.add( 1.0 );
		else output.add( 0.0 );
		
		return output;
	}

	@SuppressWarnings( "serial" )
	public List<NetworkData> setupTrainingSet() {

		List<NetworkData> trainingSet = new ArrayList<NetworkData>();

		NetworkData input1 = new NetworkData();
		input1.setInput( new ArrayList<Double>() {
			{
				add( 0.0 );
				add( 0.0 );
			}
		} );
		input1.setOutput( new ArrayList<Double>() {{ add(0.0); }} );
		trainingSet.add( input1 );
		
		NetworkData input2 = new NetworkData();
		input2.setInput( new ArrayList<Double>() {
			{
				add( 0.0 );
				add( 1.0 );
			}
		} );
		input2.setOutput( new ArrayList<Double>() {{ add(1.0); }} );
		trainingSet.add( input2 );
		
		NetworkData input3 = new NetworkData();
		input3.setInput( new ArrayList<Double>() {
			{
				add( 1.0 );
				add( 0.0 );
			}
		} );
		input3.setOutput( new ArrayList<Double>() {{ add(1.0); }} );
		trainingSet.add( input3 );
		
		NetworkData input4 = new NetworkData();
		input4.setInput( new ArrayList<Double>() {
			{
				add( 1.0 );
				add( 1.0 );
			}
		} );
		input4.setOutput( new ArrayList<Double>() {{ add(1.0); }} );
		trainingSet.add( input4 );

		return trainingSet;
	}

}
