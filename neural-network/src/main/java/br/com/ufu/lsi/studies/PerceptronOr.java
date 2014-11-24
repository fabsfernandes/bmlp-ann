
package br.com.ufu.lsi.studies;

/**
 * Using Neuroph API
 */

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.SupervisedLearning;
import org.neuroph.nnet.Perceptron;

public class PerceptronOr {

    public static void main( String ... args ) {
        
        NeuralNetwork<SupervisedLearning> neuralNetwork = new Perceptron(2, 1);
   
        DataSet trainingSet = new DataSet(2, 1); 
        trainingSet.addRow (new DataSetRow (new double[]{0, 0}, new double[]{0})); 
        trainingSet.addRow (new DataSetRow (new double[]{0, 1}, new double[]{1})); 
        trainingSet.addRow (new DataSetRow (new double[]{1, 0}, new double[]{1})); 
        trainingSet.addRow (new DataSetRow (new double[]{1, 1}, new double[]{1})); 
     
        neuralNetwork.learn(trainingSet); 

        //neuralNetwork.save(“or_perceptron.nnet”); 
        
        //NeuralNetwork neuralNetwork = NeuralNetwork.load(“or_perceptron.nnet”); 
     
        neuralNetwork.setInput(0, 0); 
 
        neuralNetwork.calculate(); 
        
        double[] networkOutput = neuralNetwork.getOutput();
        
        printOutput( networkOutput );
        
        Double[] weightsOutput = neuralNetwork.getWeights();
        
        printOutput( weightsOutput );
        
    }
    
    public static void printOutput( double[] networkOutput ) {
        
        for( double d : networkOutput ){
            System.out.println( d );
        }
    }
    
    public static void printOutput( Double[] networkOutput ) {
        
        for( double d : networkOutput ){
            System.out.println( d );
        }
    }
}
