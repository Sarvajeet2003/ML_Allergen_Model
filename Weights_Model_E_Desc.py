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

# Duplicate the E3 feature to give it more weightage
X_features['E3_duplicated'] = X_features['E3']  # Duplicate E3 feature

X = pd.concat([X_sequences, X_features], axis=1)  # Combine sequences and features

y = data['E11']  # Target variable

# One-hot encoding for amino acid sequences (example)
# You may need to implement a more sophisticated encoding method
X_encoded = pd.get_dummies(X['Protein_sequence'].apply(list).apply(pd.Series).stack()).groupby(level=0).sum()

# Combine one-hot encoded sequences with additional features
X_encoded = pd.concat([X_encoded, X_features.reset_index(drop=True)], axis=1)

# Step 3: Model Selection and Training
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest Classifier
# Check unique values in y to determine classes for class weights
classes = y.unique()

# Increase the weight of the duplicated E3 feature by setting class_weight parameter
class_weights = {"E3": 2}  # You can adjust the weight according to your preference

# Ensure classes are in class_weights
class_weights = {k: v for k, v in class_weights.items() if k in classes}

clf = RandomForestClassifier(random_state=42, class_weight=class_weights)
clf.fit(X_train_scaled, y_train)

# Step 4: Model Evaluation
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print(classification_report(y_test, y_pred))
