package br.com.ufu.lsi.studies.mlp;

import java.util.ArrayList;
import java.util.List;

import br.com.ufu.lsi.studies.model.Layer;
import br.com.ufu.lsi.studies.model.NetworkData;
import br.com.ufu.lsi.studies.model.Neuron;
import br.com.ufu.lsi.studies.util.RandomGenerator;

public class BackpropagationMLP {

	private static final Double LEARNING_RATE = 0.3;
	// private static final Double INITIAL_BIASES = 0.9;
	// private static final Double INITIAL_WEIGHTS = 0.05;
	private static final int EPOCHS = 2000;
	// private static final Double ERROR_THRESHOLD = 1.0;
	private static final Double MOMENTUM = 0.8;
	private static final Double MIN_RANGE_WEIGHT = 0.0;
	private static final Double MAX_RANGE_WEIGHT = 1.0;
	private static final Double MIN_RANGE_BIAS = -0.5;
	private static final Double MAX_RANGE_BIAS = 0.5;

	private List<Layer> layers;

	public BackpropagationMLP( Integer... args ) {

		this.layers = new ArrayList<Layer>();
		for ( int i = 0; i < args.length; i++ ) {
			Layer layer = new Layer( args[i] );
			this.layers.add( layer );
		}
	}

	public static void main( String... args ) {

		BackpropagationMLP mlp = new BackpropagationMLP( 2, 3, 1 );

		// build input set
		List<NetworkData> data = mlp.setupTrainingSet();

		// build model
		mlp.buildModel( data );

		// test model
		mlp.testNetwork( data );
	}

	public void buildModel( List<NetworkData> data ) {

		initializeParameters( data.get( 0 ) );

		trainNetwork( data );
	}

	public void testNetwork( List<NetworkData> data ) {

		for ( NetworkData sample : data ) {

			forward( sample );

			Double obtained = layers.get( layers.size() - 1 ).getNeurons()
					.get( 0 ).getActivationValue();
			for ( Double d : sample.getInput() )
				System.out.print( d + "\t" );
			System.out.print( "= " + sample.getOutput().get( 0 ) + "\t"
					+ obtained );
			System.out.println();
		}
	}

	public void trainNetwork( List<NetworkData> data ) {

		Double currentGlobalError = 0.0;
		Double previousGlobalError = 0.0;
		int epoch = 0;
		do {

			previousGlobalError = currentGlobalError;
			// adjust weights
			for ( NetworkData sample : data ) {

				forward( sample );

				// printNeuralNetwork( epoch, previousGlobalError,
				// currentGlobalError );

				backward( sample );

				// printNeuralNetwork( epoch, previousGlobalError,
				// currentGlobalError );

			}

			// calculate global error
			for ( NetworkData sample : data ) {
				forward( sample );
				calculateSampleError( sample );
			}
			currentGlobalError = calculateGlobalError( data );

			epoch++;

			printError( currentGlobalError, epoch );

		} // while ( Math.abs( currentGlobalError - previousGlobalError ) >
			// ERROR_THRESHOLD && epoch < EPOCHS );
		while ( epoch < EPOCHS );
		// while ( Math.abs( currentGlobalError - previousGlobalError ) <=
		// ERROR_THRESHOLD );
	}

	public void printNeuralNetwork( int epoch, double previousGlobalError,
			double currentGlobalError ) {

		System.out.println( "===== EPOCH " + epoch + "=====" );

		for ( int i = 0; i < layers.size(); i++ ) {
			Layer layer = layers.get( i );
			System.out.println( "## Layer " + (i + 1) );
			for ( int j = 0; j < layer.getNeurons().size(); j++ ) {
				Neuron neuron = layer.getNeurons().get( j );
				System.out.println( "\tNeuron " + (j + 1) + ": " );
				System.out.println( "\t\t activation value:"
						+ neuron.getActivationValue() );
				System.out.println( "\t\t bias:" + neuron.getBias() );
				System.out.println( "\t\t delta:" + neuron.getDelta() );
				System.out.println( "\t\t sum value:" + neuron.getSumValue() );
				System.out.print( "\t\t weights:" );
				for ( Double w : neuron.getWeights() )
					System.out.print( w + "," );
				System.out.println();
			}
		}
		System.out.println( "ERROR = " + currentGlobalError );
		System.out.println();

	}

	public void printError( Double error, int epoch ) {
		System.out.println( epoch + "\t" + error );
	}

	public void calculateSampleError( NetworkData sample ) {

		Layer output = layers.get( layers.size() - 1 );
		Double sum = 0.0;
		for ( int i = 0; i < output.getNeurons().size(); i++ ) {
			Neuron neuron = output.getNeurons().get( i );
			Double obtainedValue = neuron.getActivationValue();
			Double desiredValue = sample.getOutput().get( i );
			sum += Math.pow( desiredValue - obtainedValue, 2.0 );
		}
		sample.setError( sum / 0.5 );
	}

	public Double calculateGlobalError( List<NetworkData> data ) {
		Double sum = 0.0;
		for ( NetworkData sample : data ) {
			sum += sample.getError();
		}
		return sum / (double) data.size();
	}

	public void backward( NetworkData sample ) {

		for ( int i = layers.size() - 1; i > 0; i-- ) {

			Layer layer = layers.get( i );
			Layer previousLayer = layers.get( i - 1 );

			if ( i == layers.size() - 1 ) {
				calculateDeltaOutputLayer( layer, sample );

			} else {
				Layer nextLayer = layers.get( i + 1 );
				calculateDeltaHiddenLayers( layer, nextLayer );
			}
			adjustWeights( layer, previousLayer );
		}

	}

	public void calculateDeltaOutputLayer( Layer outpuLayer, NetworkData sample ) {

		for ( int i = 0; i < outpuLayer.getNeurons().size(); i++ ) {
			Neuron neuron = outpuLayer.getNeurons().get( i );
			Double desired = sample.getOutput().get( i );
			Double obtained = neuron.getActivationValue();
			Double derivative = derivativeFunction( neuron.getSumValue() );
			// Double derivative = derivativeFunction(
			// neuron.getActivationValue() );

			Double delta = (desired - obtained) * derivative;

			neuron.setDelta( delta );
		}
	}

	public void calculateDeltaHiddenLayers( Layer currentLayer, Layer nextLayer ) {

		for ( int i = 0; i < currentLayer.getNeurons().size(); i++ ) {
			Neuron neuron = currentLayer.getNeurons().get( i );

			Double sum = 0.0;
			for ( int j = 0; j < neuron.getWeights().size(); j++ ) {
				Double deltaNextLayer = nextLayer.getNeurons().get( j )
						.getDelta();
				Double weight = neuron.getWeights().get( j );
				sum += deltaNextLayer * weight;
			}

			Double derivative = derivativeFunction( neuron.getSumValue() );
			//Double derivative = derivativeFunction( neuron.getActivationValue() );

			Double delta = (sum * derivative);

			neuron.setDelta( delta );
		}

	}

	public void adjustWeights( Layer currentLayer, Layer previousLayer ) {

		for ( Neuron neuron : previousLayer.getNeurons() ) {
			for ( int i = 0; i < neuron.getWeights().size(); i++ ) {
				Double currentWeight = neuron.getWeights().get( i );
				Double currentLayerDelta = currentLayer.getNeurons().get( i )
						.getDelta();
				Double previousLayerActivationNeuronValue = neuron
						.getActivationValue();

				Double weightVariation = (LEARNING_RATE * currentLayerDelta * previousLayerActivationNeuronValue);

				Double weightPreviousVariation = neuron
						.getPreviousWeightsVariations().get( i );

				if ( weightPreviousVariation == null ) {
					weightPreviousVariation = 0.0;
				}

				Double newWeight = currentWeight + weightVariation
						+ (weightPreviousVariation * MOMENTUM);

				neuron.getWeights().set( i, newWeight );

				neuron.getPreviousWeightsVariations().set( i, weightVariation );
			}
		}
		for ( Neuron neuron : currentLayer.getNeurons() ) {
			Double currentBias = neuron.getBias();
			Double currentLayerDelta = neuron.getDelta();

			Double previousBiasVariation = neuron.getPreviousBiasVariation();
			if ( previousBiasVariation == null ) {
				previousBiasVariation = 0.0;
			}

			Double biasVariation = LEARNING_RATE * currentLayerDelta;
			Double newBias = currentBias + biasVariation
					+ (previousBiasVariation * MOMENTUM);
			neuron.setBias( newBias );
			neuron.setPreviousBiasVariation( biasVariation );
		}
	}

	public void forward( NetworkData sample ) {

		for ( int i = 0; i < layers.size(); i++ ) {

			Layer layer = layers.get( i );

			if ( i == 0 ) {
				layer.setNeuronsActivationValues( sample.getInput() );

			} else {
				calculateNeuronsValues( layer, layers.get( i - 1 ) );
			}
		}
	}

	public void calculateNeuronsValues( Layer currentLayer, Layer previousLayer ) {

		for ( int i = 0; i < currentLayer.getNeurons().size(); i++ ) {
			Neuron neuron = currentLayer.getNeurons().get( i );

			Double sum = 0.0;
			for ( Neuron previousNeuron : previousLayer.getNeurons() ) {
				Double weight = previousNeuron.getWeights().get( i );
				Double value = previousNeuron.getActivationValue();
				sum += (weight * value);
			}
			sum += neuron.getBias();
			neuron.setSumValue( sum );

			Double activation = activationFunction( sum );
			neuron.setActivationValue( activation );
		}

	}

	public Double derivativeFunction( Double value ) {

		Double result;

		// tanh'
		//result = Math.pow( 2.0 / (Math.exp( value ) + Math.exp( -value )),
		//2.0 );
		result = 1 - Math.pow(activationFunction( value ),2);

		// sigmoid'
		// result = Math.exp( value ) / ( Math.pow( Math.exp( value ) + 1, 2 )
		// );

		// sigmoid'
		//result = activationFunction( value )
		//		* (1 - activationFunction( value ));

		return result;
	}

	public Double activationFunction( Double sum ) {

		Double result;

		// tanh
		result = Math.tanh( sum );
		//result = (double) Math.round( result );

		// sigmoid
		//result = 1.0 / (1.0 + Math.exp( -sum ));

		return result;
	}

	public void initializeParameters( NetworkData data ) {

		for ( int i = 0; i < layers.size() - 1; i++ ) {

			Layer layer = layers.get( i );
			int nextLayerSize = layers.get( i + 1 ).getNeurons().size();

			// for ( Neuron neuron : layer.getNeurons() ) {
			for ( int k = 0; k < layer.getNeurons().size(); k++ ) {
				Neuron neuron = new Neuron();
				neuron.setPreviousWeightsVariations( new ArrayList<Double>() );
				neuron.setWeights( new ArrayList<Double>() );
				for ( int j = 0; j < nextLayerSize; j++ ) {
					neuron.getWeights().add(
							RandomGenerator.randDouble( MIN_RANGE_WEIGHT,
									MAX_RANGE_WEIGHT ) );
					neuron.getPreviousWeightsVariations().add( null );
				}
				neuron.setBias( RandomGenerator.randDouble( MIN_RANGE_BIAS,
						MAX_RANGE_BIAS ) );
				layer.getNeurons().set( k, neuron );
			}
		}

		// last layer
		Layer lastLayer = layers.get( layers.size() - 1 );
		for ( int i = 0; i < lastLayer.getNeurons().size(); i++ ) {
			Neuron neuron = new Neuron();
			neuron.setPreviousWeightsVariations( new ArrayList<Double>() );
			neuron.setWeights( new ArrayList<Double>() );
			neuron.getWeights().add(
					RandomGenerator.randDouble( MIN_RANGE_WEIGHT,
							MAX_RANGE_WEIGHT ) );
			neuron.getPreviousWeightsVariations().add( null );
			neuron.setBias( RandomGenerator.randDouble( MIN_RANGE_BIAS,
					MAX_RANGE_BIAS ) );
			lastLayer.getNeurons().set( i, neuron );
		}
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
		input1.setOutput( new ArrayList<Double>() {
			{
				add( 0.0 );
			}
		} );
		trainingSet.add( input1 );

		NetworkData input2 = new NetworkData();
		input2.setInput( new ArrayList<Double>() {
			{
				add( 0.0 );
				add( 1.0 );
			}
		} );
		input2.setOutput( new ArrayList<Double>() {
			{
				add( 1.0 );
			}
		} );
		trainingSet.add( input2 );

		NetworkData input3 = new NetworkData();
		input3.setInput( new ArrayList<Double>() {
			{
				add( 1.0 );
				add( 0.0 );
			}
		} );
		input3.setOutput( new ArrayList<Double>() {
			{
				add( 1.0 );
			}
		} );
		trainingSet.add( input3 );

		NetworkData input4 = new NetworkData();
		input4.setInput( new ArrayList<Double>() {
			{
				add( 1.0 );
				add( 1.0 );
			}
		} );
		input4.setOutput( new ArrayList<Double>() {
			{
				add( 0.0 );
			}
		} );
		trainingSet.add( input4 );

		return trainingSet;
	}

}
