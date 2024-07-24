import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
# Initialize lists to store overfit or underfit percentages
overfit_percentages = []
underfit_percentages = []
# Initialize lists to store training and validation losses for all configurations
all_train_losses = []
all_val_losses = []
# Step 1: Read data
data = pd.read_csv('/Users/sarvajeethuk/Downloads/IP/Final/E_Descriptors_Final.csv')

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
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert target labels to one-hot encoding
num_classes = len(np.unique(y))
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)

# Define hyperparameters to iterate over
epochs_list = [10, 20, 30]
optimizer_list = ['adam', 'sgd']
activation_list = ['relu', 'tanh']


# Iterate over hyperparameters
# for epochs in epochs_list:
#     for optimizer in optimizer_list:
#         for activation in activation_list:
#             # Initialize the model
#             model = tf.keras.Sequential([
#                 tf.keras.layers.Dense(64, activation=activation, input_shape=(X_train_scaled.shape[1],)),
#                 tf.keras.layers.Dense(32, activation=activation),
#                 tf.keras.layers.Dense(num_classes, activation='softmax')
#             ])
            
#             # Compile the model
#             model.compile(optimizer=optimizer,
#                           loss='categorical_crossentropy',
#                           metrics=['accuracy'])
            
#             # Train the model
#             history = model.fit(X_train_scaled, y_train_one_hot, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
        
#             # Evaluate the model
#             _, accuracy = model.evaluate(X_test_scaled, y_test_one_hot, verbose=0)
#             train_accuracy = history.history['accuracy'][-1]
#             val_accuracy = history.history['val_accuracy'][-1]
            
#             # Calculate overfit or underfit percentage
#             difference = train_accuracy - val_accuracy
#             if difference > 0:
#                 overfit_percentage = (difference / train_accuracy) * 100
#                 overfit_percentages.append(overfit_percentage)
#             elif difference < 0:
#                 underfit_percentage = (-difference / train_accuracy) * 100
#                 underfit_percentages.append(underfit_percentage)

#             print(f'Epochs: {epochs}, Optimizer: {optimizer}, Activation: {activation}, Test Accuracy: {accuracy * 100:.2f}%')

#             # # Plot training & validation accuracy values
#             # plt.plot(history.history['accuracy'])
#             # plt.plot(history.history['val_accuracy'])
#             # plt.title('Model accuracy')
#             # plt.ylabel('Accuracy')
#             # plt.xlabel('Epoch')
#             # plt.legend(['Train', 'Validation'], loc='upper left')
#             # plt.show()

#             # # Plot training & validation loss values
#             # plt.plot(history.history['loss'])
#             # plt.plot(history.history['val_loss'])
#             # plt.title('Model loss')
#             # plt.ylabel('Loss')
#             # plt.xlabel('Epoch')
#             # plt.legend(['Train', 'Validation'], loc='upper left')
#             # plt.show()

# # Calculate average overfit or underfit percentage
# avg_overfit_percentage = np.mean(overfit_percentages) if overfit_percentages else 0
# avg_underfit_percentage = np.mean(underfit_percentages) if underfit_percentages else 0

# print(f'Average Overfit Percentage: {avg_overfit_percentage:.2f}%')
# print(f'Average Underfit Percentage: {avg_underfit_percentage:.2f}%')



import numpy as np

# Initialize lists to store training and validation losses for all configurations
all_train_losses = []
all_val_losses = []

# Iterate over hyperparameters
for epochs in epochs_list:
    for optimizer in optimizer_list:
        for activation in activation_list:
            # Initialize the model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation=activation, input_shape=(X_train_scaled.shape[1],)),
                tf.keras.layers.Dense(32, activation=activation),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            # Compile the model
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            # Train the model
            history = model.fit(X_train_scaled, y_train_one_hot, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
            
            # Append training and validation losses to the lists
            all_train_losses.append(history.history['loss'])
            all_val_losses.append(history.history['val_loss'])

# Find the maximum length of loss lists
max_length = max(len(loss_list) for loss_list in all_train_losses + all_val_losses)

# Pad the shorter lists with zeros
all_train_losses = [loss_list + [0] * (max_length - len(loss_list)) for loss_list in all_train_losses]
all_val_losses = [loss_list + [0] * (max_length - len(loss_list)) for loss_list in all_val_losses]

# Convert lists to NumPy arrays
all_train_losses = np.array(all_train_losses)
all_val_losses = np.array(all_val_losses)

# Calculate average training and validation losses
avg_train_loss = np.mean(all_train_losses, axis=0)
avg_val_loss = np.mean(all_val_losses, axis=0)

# Plot average training & validation loss values
plt.plot(avg_train_loss, label='Average Training Loss')
plt.plot(avg_val_loss, label='Average Validation Loss')
plt.title('Average Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()




from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get predictions from the model
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()