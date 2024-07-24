import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Read data
data = pd.read_csv('/Users/sarvajeethuk/Downloads/E_Descriptors_Final.csv')

# Step 2: Data Preprocessing
X_sequences = data['Protein_sequence']  # Amino acid sequences
X_features = data[['E1', 'E2', 'E3', 'E4', 'E5', 'E9', 'E10', 'E6', 'E7', 'E8']]  # Additional features
X = pd.concat([X_sequences, X_features], axis=1)  # Combine sequences and features

y = data['E11']  # Target variable

# One-hot encoding for amino acid sequences (example)
# You may need to implement a more sophisticated encoding method
X_encoded = pd.get_dummies(X['Protein_sequence'].apply(list).apply(pd.Series).stack()).groupby(level=0).sum()

# Combine one-hot encoded sequences with additional features
X_encoded = pd.concat([X_encoded, X_features.reset_index(drop=True)], axis=1)

# Step 3: Model Selection and Training
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_scaled, y)

# Step 4: Predict allergenicity
# Predict probabilities of being allergic (class 1)
y_prob_allergic = clf.predict_proba(X_scaled)[:, 1]

# Convert predicted probabilities to percentages
y_percentage_allergic = y_prob_allergic * 100

# Add the predicted allergenic percentage as a new column in the DataFrame
data['Predicted_Allergenic_Percentage'] = y_percentage_allergic

# Save the DataFrame with the new column to a new CSV file
data.to_csv('/Users/sarvajeethuk/Downloads/E_Descriptors_Final_with_Predictions.csv', index=False)
