Mary Beth Hudson
Created April 1, 2020
Project for CSCI - 4336 Machine Learning
Instructor: Abukmail
School: University of Houston - Clear Lake


Purpose:
	This project uses the Logistic Regression Algorithm to recognize heart arrhythmias in 
	actual patient EKG data that came from the Kaggle database. The domain and the algorithm
 	were new to me so I also documented some basic information about heart arrhythmias which
	captures the necessity for developing this type of technology. 

	Below are the steps I took and the files used in each step. To test out the Logistic
 	Regression Algorithm, I also ran it on other data sets that I was more familiar with
 	(such as the Iris dataset) just to be sure it was generic enough and generally worked as
 	expected. The actual parameters that I used for the heart arrhythmia recognition are 
	documented in the program. The user can run it using these parameters or select their own.

	See Step 4 to run the program.


Steps taken to learn how to recognize heart arrhythmias:

	Step 1	Understanding Arrhythmia Data 
	      	file: Understanding Arrhythmias.docx – Overview of Arrhythmia data

	Step 2	Selecting the Arrhythmia Dataset
		file: Selecting the Dataset.docx – Looking at possible data sets to use
		file: ExamineDatasets.ipynb - Visualizes several data sets before selecting one
			                      Uses Jupyter Notebooks to analyze/visualize the data
		file: rawDatasets – folder containing all the datasets I visualized

	Step 3 	Cleaning the Arrhythmia data
		file: DataSet2.ipynb - Jupyter Notebook file showing how to analyze and clean
			the Kaggle data. Running this file will create the two data files needed 
			for the java program:
        			      Arrhythmia_TrainingData.csv - the cleaned data to train the LGD 
        			      Arrhythmia_TestingData.csv - the cleaned data to test the LGD
       		file: data_arrhythmia.csv - the selected database from Kaggle
		file: PCA on Arrhythmia data.ipynb - running Principle Component Analysis 
			on the data
        		

	Step 4 	Running the Program
		file: Running the Program.docx – Instructions for running the Logistic Regression 
			Algorithm. It will run in batch mode (preselected parameters) or
 			interactive mode (user selects the parameter and how long to run it). 
			It runs using a Stochastic Gradient Descent or Batch Gradient Descent.


	NOTE:	Other Data Sets that the Logistic Regression Algorithm has run on:
		(Uncomment them out in main procedure to run them)
			IrisTrainingData.csv/IrisTestingData.csv
			HeartTrainingData.csv/HeartTestingData.csv
			Arrhythmia_First_TrainingData.csv/ Arrhythmia_First_TrainingData.csv
			Arrhythmia_Last_TrainingData.csv/ Arrhythmia_Last_TrainingData.csv





