import java.util.Arrays;
import java.util.ArrayList;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;
import java.util.Random; 

/*
 * Mary Beth McMahon Hudson - Logistic Regression with Stochastic Gradient Descent
 */
public class LGD {
    private double[] weightVector;
    private double learningRate;
    private int maxNumOfRuns;
    private int numFeatures;
    private double allowableError;
    private boolean showSummary;
    private boolean showDetails;
    private double randomnessVar;
    private boolean seeVectorInfo = false;
	private Random rand = new Random();
	
        
	public LGD(double rate, int iterations, double amtOfError, double randomness, boolean details, boolean summary, int numberOfFeatures) {
		weightVector = new double[numberOfFeatures];
 		learningRate = rate;
 		allowableError = amtOfError;
    	maxNumOfRuns = iterations;
    	numFeatures = numberOfFeatures;
    	randomnessVar = randomness;
    	showSummary = summary;
    	showDetails = details;
    	rand.setSeed(44);
	}

	public static List<Sample> readTheDataset(String file) throws FileNotFoundException, IllegalArgumentException {
		List<Sample> dataset = new ArrayList<Sample>();
		Scanner scanner = null;
		//int linenum = 0; 
		try {
			scanner = new Scanner(new File(file));
			while(scanner.hasNextLine()) {

				String line = scanner.nextLine();
				if (line.startsWith("#")) {
					continue;
				}
				String[] columns = line.split(",");

				if (dataset.size() > 0){
					if (columns.length != dataset.get(0).featureVector.length + 1) {
		    			throw new IllegalArgumentException("Wrong number of features in dataset " + file);
		    		}
		    	}

				int i = 0;
				double[] data = new double[columns.length-1];
				for (i=0; i<columns.length-1; i++) {
					data[i] = Double.parseDouble(columns[i]);
				}
				int label = Integer.parseInt(columns[i]);
				Sample sample = new Sample(data, label);
				dataset.add(sample);
			}
		} finally {
			if (scanner != null)
				scanner.close();
		}
		return dataset;
	}

	public void trainOnTheDataset(List<Sample> sampleSet) {
		for (int n=0; n<maxNumOfRuns; n++) {
			double greatestAbsError = 0.0; // for batch gradient descent
			double greatestCurrentError = 0.0; // actual error for batch gradient descent
			double likelihood = 0.0;
			double[] imposingVector = sampleSet.get(0).featureVector;
			int imposingIndex = 0; // for batch gradient descent
			for (int i=0; i<sampleSet.size(); i++) {
				double[] featureVector = sampleSet.get(i).featureVector;
				double yhat = predictProbability(featureVector);
				int label = sampleSet.get(i).label;

				double currentError =  label - yhat;
				double absError =  Math.abs(currentError);
				boolean good = absError <= allowableError;
				if (seeVectorInfo && showDetails && good) {
					System.out.println("good enough: " + good);
				}

				// Stochastic Gradient Descent
				if (Math.abs(yhat - label) > allowableError) {
					// System.out.println("label " + label + " yhat " + yhat + " CurrErr " +currentError);
					// Batch Gradient Descent - save info for end of iteration
					if (randomnessVar > 1.0) { 
						if (absError > greatestAbsError) {
							greatestAbsError = absError;
							greatestCurrentError = currentError;
							imposingVector = featureVector;
							imposingIndex = i;
						}
							
					} else {
						double chance = rand.nextDouble();
						if (randomnessVar >= chance) {
							if (seeVectorInfo) {
								System.out.println("updating weight based on vector " + i + " rand:" + chance);
							}
							for (int j=0; j<weightVector.length; j++) {
								double error = label - yhat;
								double stepSize = learningRate * error;
								weightVector[j] = weightVector[j] + stepSize * featureVector[j];
							}
							likelihood += label * Math.log(predictProbability(featureVector)) + (1-label) * Math.log(1- predictProbability(featureVector));
						} else {
							if (seeVectorInfo) {
								System.out.println("skipping " + i + " rand:" + chance);
							}
						}
					}
				}
			}
			if (randomnessVar > 1.0) {
				// System.out.println("Iteration " + n + ":  item " + imposingIndex  + " with error: " + greatestCurrentError); 
				for (int j=0; j<weightVector.length; j++) {
					double stepSize = learningRate * greatestCurrentError;
					weightVector[j] = weightVector[j] + stepSize * imposingVector[j];
				}
			}
			if (showDetails) {
				System.out.println("iteration: " + n + " likelihood: " + likelihood);
			}
			if (showDetails) { 
				System.out.println("Final Vector: " + Arrays.toString(weightVector)); 
			}
		}
	}

	public void predictTheTestSet(List<Sample> testSet) {
		int correct = 0;
		int incorrect = 0;
		double totalError = 0.0;
		for (int i=0; i<testSet.size(); i++) {
			double[] featureVector = testSet.get(i).featureVector;
			int label = testSet.get(i).label;
			double yhat = predictProbability(featureVector);
			double absError = Math.abs(label - yhat);
			
			if (showSummary) {
				if (absError <= allowableError) {
					correct++;
				} else {
					incorrect++;
					totalError += absError - allowableError;
				}
			}
			if (showDetails) {
				System.out.println("predicting " + yhat + " for " + label);
				if (absError < allowableError) {
					System.out.print("CORRECT ");
				} else {
					System.out.print("INCORRECT ");
				}
				if (label == 0) {
					System.out.print("Negative result - ");
				} else {
					System.out.print("Positive result - ");
				}
				System.out.println("Probability = " + yhat);
			}
		}
		double total = testSet.size();
		double percentCorrect = (correct/total) * 100.0;
		double percentError = (totalError/total) * 100.0;
		if (showSummary) {
			String learningRateStr = String.format("%.6f", learningRate); 
			System.out.print("LR: " + learningRateStr);
			String allowableErrorStr = String.format("%1.4f", allowableError); 
			System.out.print(" E: " + allowableErrorStr);
			String randomnessVarStr = String.format("%1.4f", randomnessVar); 
			System.out.print(" randomness var: " + randomnessVarStr);
			String maxNumOfRunsStr = String.format("%8d", maxNumOfRuns);
			System.out.print(" Num Iterations: " + maxNumOfRunsStr);
			System.out.print("    *** Total Correct: " + correct);
			System.out.print("   Total Incorrect: " + incorrect);
			System.out.print("    Percent correct " + percentCorrect);
			System.out.println();

		}
		if (!showSummary) {
			System.out.println("    Percent correct " + percentCorrect);
		}
	}

	private double predictProbability(double[] featureVector) {
		// System.out.println("weightVector: " + Arrays.toString(weightVector) + "featureVector: " + Arrays.toString(featureVector)); 
		return sigmoidFunction(dotProduct(weightVector,featureVector));
	} 

	private static double dotProduct(double[] w, double[] x) {
		double sum = 0.0;
		for (int i=0; i<w.length;i++)  {
			sum += w[i] * x[i];
		}
		return sum;
	}

	private static double sigmoidFunction(double s) {
		return 1.0 / (1.0 + Math.exp(-s));
	}

	public static class Sample {
		public int label;
		public double[] featureVector;

		public Sample(double[] featureVector, int label) {
			this.featureVector = featureVector;
			this.label = label;
		}
	}

	private double e_exponent (double[] featureVector, int label, double yhat) {
		return -1.0 * yhat * dotProduct(weightVector,featureVector);
	}

	private double naturalLogFunctionForLogisticRegression (double[] featureVector, int label, double yhat) {
		double exponentVal = e_exponent(featureVector, label, yhat);
		return Math.log(1.0 + Math.exp(exponentVal));
	}


	public void CalulateError(List<Sample> testSet, String errorLabel) {
		double sum = 0.0;
		for (int i=0; i<testSet.size(); i++) {
			double[] featureVector = testSet.get(i).featureVector;
			int label = testSet.get(i).label;
			double yhat = predictProbability(featureVector);
			sum += naturalLogFunctionForLogisticRegression(featureVector, label, yhat);
		}
		double error = sum/testSet.size();
		if (showSummary) {
			System.out.println (errorLabel + " " + error);	
		}
	}

	public static void main(String[] args) throws FileNotFoundException {
		boolean details = false;
		boolean summary = true;
		boolean showEinEout = false;
		boolean promptUser = false;
		int repetitionRate = 1; // to run same settings more than once
		String trainingFilename = "Arrhythmia_TrainingData.csv";
		String testingFilename = "Arrhythmia_TestingData.csv";
		//trainingFilename = "IrisTrainingData.csv";
		//testingFilename = "IrisTestingData.csv";
		//trainingFilename = "HeartTrainingData.csv";
		//testingFilename = "HeartTestingData.csv";
		//trainingFilename = "Arrhythmia_First_TrainingData.csv";
		//testingFilename = "Arrhythmia_First_TestingData.csv";
		//trainingFilename = "Arrhythmia_Last_TrainingData.csv";
		//testingFilename = "Arrhythmia_Last_TestingData.csv";

		List<Sample> sampleSet = readTheDataset(trainingFilename);
		List<Sample> testSet = readTheDataset(testingFilename);
		int numberOfFeatures = sampleSet.get(0).featureVector.length;
		int trainSetSize = sampleSet.size();
		int testSetSize = testSet.size();

		/* String[] learningRate_inputs = new String[]{"0.1"};
		String[] numIterations_inputs = new String[]{"1000"};
		String[] allowableError_inputs = new String[]{"0.01"};
		String[] randomnessOfSGD_inputs = new String[]{"0.01"};
		*/

		String[] learningRate_inputs = new String[]{"0.1", "0.01", "0.001"};
		String[] numIterations_inputs = new String[]{"100", "1000"};
		String[] allowableError_inputs = new String[]{"0.1", "0.01"};
		String[] randomnessOfSGD_inputs = new String[]{"0.1", "0.01"}; 

		Scanner scanner = new Scanner(System.in);
		System.out.println("Interactive Mode? (y or n)");
		String response = scanner.nextLine();
		if (response.contains("y") || response.contains("Y")) {
			promptUser = true;
		}

		if (promptUser) {
			System.out.println("Enter learning rates (float -> e.g. 0.01) (space separated)");
			learningRate_inputs = scanner.nextLine().split(" ");
			System.out.println("Enter allowable error (float -> e.g. 0.1) (space separated)");
			allowableError_inputs = scanner.nextLine().split(" ");
			System.out.println("Enter randomness numbers (0.0 - 1.0) for SGD or (2.0) for Batch Gradient Descent (float -> e.g. 0.01) (space separated)");
			randomnessOfSGD_inputs = scanner.nextLine().split(" ");
			System.out.println("Enter max number of iterations (int -> e.g. 1000) (space separated)");
			numIterations_inputs = scanner.nextLine().split(" ");
		}

		System.out.println("Running Logistic Regression SGD on dataset: " + trainingFilename
			                + " training set size " + trainSetSize + " test set size " + testSetSize);
		System.out.println("");
		for (int reps = 0; reps < repetitionRate; reps++){
			for(int lr = 0; lr < learningRate_inputs.length; lr++){
				double learningRate = Double.parseDouble(learningRate_inputs[lr]);
				for(int ae = 0; ae < allowableError_inputs.length; ae++){
					double allowableError = Double.parseDouble(allowableError_inputs[ae]);
					for(int r = 0; r < randomnessOfSGD_inputs.length; r++){
						double randomness = Double.parseDouble(randomnessOfSGD_inputs[r]);
						for(int itr = 0; itr < numIterations_inputs.length; itr++){
							int iterations = Integer.parseInt(numIterations_inputs[itr]);
							if (!summary) {
								System.out.println("");
								System.out.print("*** Running LRGD with learning rate as " + learningRate);
								System.out.print(" allowabeError rate as " + allowableError);
								System.out.println(" iterations as " + iterations + " ***");
								System.out.println("");
							}
							LGD LogisticRegressionWithGradientDescent = new LGD(learningRate, iterations, allowableError, randomness, details, summary, numberOfFeatures);
							LogisticRegressionWithGradientDescent.trainOnTheDataset(sampleSet);
							LogisticRegressionWithGradientDescent.predictTheTestSet(testSet);
							if (showEinEout) {
								LogisticRegressionWithGradientDescent.CalulateError(sampleSet, "Error(in)");
								LogisticRegressionWithGradientDescent.CalulateError(testSet, "Error(out)");
							}
						}
					}
				}
			}
		}
	}
}
