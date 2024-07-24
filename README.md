**IP\_Winter\_ML&DL.py**

This Python script contains implementations of various machine learning (ML)and deep learning (DL)models for classification tasks. Below is a brief overview of each model implementation along with instructions on how to use them:

**RandomForestClassifier**

This section of the script demonstrates the usage of the RandomForestClassifier from scikit-learn for classification tasks. It trains a Random Forest model on a dataset and evaluates its performance.

**Usage:**

1. Ensure you have Python installed along with the required libraries: numpy, scikit-learn.
1. Replace final\_train with your dataset.
1. Execute the script.

**LogisticRegression**

This section of the script demonstrates the usage of the LogisticRegression model from scikit-learn for classification tasks. It trains a logistic regression model on a dataset and evaluates its performance.

**Usage:**

1. Ensure you have Python installed along with the required libraries: numpy, scikit-learn.
1. Replace final\_train with your dataset.
1. Execute the script.

**Deep Learning**

This section of the script demonstrates the usage of a simple deep learning model using TensorFlow and Keras for classification tasks. It trains a neural network on a dataset and evaluates its performance.

**Usage:**

1. Ensure you have Python installed along with the required libraries: numpy, scikit-learn, tensorflow, keras.
1. Replace final\_train with your dataset.
1. Execute the script.

**Hyper-ParameterTuning Algorithm**

This section of the script demonstrates the usage of hyper-parameter tuning for a neural network model using TensorFlow and Keras. It iterates over different combinations of hyperparameters to find the optimal model.

**Usage:**

1. Ensure you have Python installed along with the required libraries: numpy, scikit-learn, tensorflow, keras.
1. Replace final\_train with your dataset.
1. Execute the script.

**Sequential Neural Network Algorithm DL**

This section of the script demonstrates the usage of a more complex neural network model using TensorFlow and Keras for classification tasks. It trains a sequential neural network on a dataset and evaluates its performance.

**Usage:**

1. Ensure you have Python installed along with the required libraries: numpy, scikit-learn, tensorflow, keras.
1. Replace final\_train with your dataset.
1. Execute the script.

**Protein Descriptors Calculation**

This Python script calculates various descriptors for protein sequences, including E descriptors, additional features, and cross-reactivity scores.

**Features:**

**EDescriptors Calculation**

The script calculates the following Edescriptors for each protein sequence:

- E1: Hydrophilic nature of peptides
- E2: Length of peptides
- E3: Tendency for helical formation
- E4: Abundance and distribution of positively charged amino acids
- E5: Tendency for β strand formation
- E11: Motif count normalized by protein sequence length
- E9: Susceptibility at low pH

**Additional Features Calculation**

Additionally,the script computes the following additional features for each protein sequence:

- E6: Number of disulfide bonds
- E7: Stability from pepsin at pH 1.0
- E8: Thermal stability based on alpha helix and beta sheet fractions

**Cross-Reactivity Score Calculation**

Furthermore, the script performs a BLASTsearch against a database of known allergens or epitopes to calculate the cross-reactivity score for each protein sequence.

**Usage:**

1. Ensure you have Python 3 installed, along with the required libraries: Biopython, pandas.
1. Replace '/Users/sarvajeethuk/Downloads/train\_Sequence.csv' with the path to your input CSVfile containing protein sequences.
1. Execute the script.

python Protein\_Descriptors\_Calculation.py

**Output:**

The script willgenerate a new CSVfile named

result\_with\_e\_descriptors\_and\_additional\_features\_Train.csv,containing the

original data along with the calculated Edescriptors, additional features, and cross-reactivity scores for each protein sequence.

**Protein Descriptor-based Machine Learning and Deep Learning Models**

This Python script implements machine learning (ML)and deep learning (DL)models using Edescriptors derived from protein sequences.

**Random Forest Classifier Usage:**

1. Ensure you have Python installed along with the required libraries: pandas, scikit-learn, tensorflow.
1. Replace '/Users/sarvajeethuk/Downloads/E\_Descriptors\_Final.csv' with the path to your input CSVfile containing Edescriptors.
1. Execute the script.

python ML\_DL\_Model\_E\_Desc.py

**Description:**

- Reads the Edescriptors from the input CSVfile.
- Preprocesses the data by combining amino acid sequences and additional features.
- Trains a Random Forest Classifier on the preprocessed data.
- Evaluates the model's performance using accuracy and classification report.

**Artificial Neural Network (ANN) Usage:**

1. Ensure you have Python installed along with the required libraries: pandas, numpy, tensorflow.
1. Replace '/Users/sarvajeethuk/Downloads/E\_Descriptors\_Final.csv' with the path to your input CSVfile containing Edescriptors.
1. Execute the script.

**Description:**

- Reads the Edescriptors from the input CSVfile.
- Preprocesses the data by combining amino acid sequences and additional features.
- Trains an Artificial Neural Network (ANN)on the preprocessed data.
- Evaluates the model's performance using accuracy and classification report.

**HyperparameterTuning Usage:**

1. Ensure you have Python installed along with the required libraries: pandas, numpy, tensorflow.
1. Replace '/Users/sarvajeethuk/Downloads/E\_Descriptors\_Final.csv' with the path to your input CSVfile containing Edescriptors.
1. Execute the script.

**Description:**

- Reads the Edescriptors from the input CSVfile.
- Preprocesses the data by combining amino acid sequences and additional features.
- Performs hyperparameter tuning for the ANNmodel.
- Evaluates and prints the test accuracy for different combinations of epochs, optimizers, and activation functions.

**Protein Allergenic Percentage Prediction**

This Python script calculates the predicted allergenic percentage of protein sequences using Random Forest Classifier based on Edescriptors.

**Usage:**

1. Ensure you have Python installed along with the required libraries: pandas, scikit-learn.
1. Replace '/Users/sarvajeethuk/Downloads/E\_Descriptors\_Final.csv' with the path to your input CSVfile containing Edescriptors.
1. Execute the script.

python Allergenic\_Percentage.py

**Description:**

- **Step 1: Read Data**: Reads the Edescriptors from the input CSVfile.
- **Step 2: Data Preprocessing**: Preprocesses the data by combining amino acid sequences and additional features, performs one-hot encoding for amino acid sequences, and combines one-hot encoded sequences with additional features.
- **Step 3: Model Selection and Training**: Scales the features, initializes and trains the Random Forest Classifier.
- **Step 4: Predict Allergenicity**: Predicts the probabilities of being allergic (class 1) for each protein sequence, converts predicted probabilities to percentages, adds the predicted allergenic percentage as a new column in the DataFrame.
- **Output**: Saves the DataFrame with the new column to a new CSVfile named E\_Descriptors\_Final\_with\_Predictions.csv.

**Random Forest Classifier with Weighted E Descriptors**

This Python script trains a Random Forest Classifier with weighted Edescriptors to predict a target variable.

**Usage:**

1. Ensure you have Python installed along with the required libraries: pandas, scikit-learn.
1. Replace '/Users/sarvajeethuk/Downloads/E\_Descriptors\_Final.csv' with the path to your input CSVfile containing Edescriptors.
1. Execute the script.

python Weights\_Model\_E\_Desc.py

**Description:**

- **Step 1: Read Data**: Reads the Edescriptors from the input CSVfile.
- **Step 2: Data Preprocessing**: Prepares the feature set by combining amino acid sequences and additional features. It duplicates the E3 feature to give it more weightage.
- **Step 3: Model Selection and Training**: Splits the data into training and testing sets, performs feature scaling, and initializes and trains the Random Forest Classifier. It assigns more weight to the duplicated E3 feature using the class\_weight parameter.
- **Step 4: Model Evaluation**: Predicts the target variable on the test set and evaluates the model's performance using accuracy and classification report.
