import numpy as np
import pandas as pd

def readData(name_dataset, path):
  data = []
  laststring = ""
  lastcount = -1
  with open(path, "r") as f:
    count = -1
    s = ""
    for line in f:
      if (line[0] == '>'):
        count += 1
        temp = [f"{name_dataset}{count}", s]
        data.append(temp)
        s = ""
      else:
        s += line.strip()
    lastcount = count + 1
    laststring = s

  data.append([f"{name_dataset}{lastcount}", laststring])
  data.pop(0)
  data = np.array(data)

  return data

"""
processing the algpred dataset into a numpy array


encoding the same data into a new column
allergen -> 1
non-allergen -> 0
algpred -> done


compare_fasta(all allergens) -> done
allertop_allergen(all allergens) -> done
allertop_nonallergen(all non-allergens) -> done
uniprot_allergen(all allergens) -> done
allergen_online(all allergens) -> done
"""

#ALGPRED DATA
df_algpred = pd.read_csv('wk2_ip_datasets/processed.csv')
algpred_data = df_algpred.values

#cleaning the algpred data to get rid of the \n values from the end
algpred_data = np.vectorize(lambda x: x.strip('\n'))(algpred_data)
condition = np.char.startswith(algpred_data[:,0].astype(str), 'A')
new_column = np.where(condition, 1, 0)
algpred_data = np.column_stack((algpred_data, new_column))
print(algpred_data.shape)
print(algpred_data)
print("\n\n\n")

#COMPARE_ALLERGEN DATA
compare_allergen_data = readData("compare_allergen", "wk2_ip_datasets/COMPARE-2023-FastA-Seq.txt")
rows_compareallergen = compare_allergen_data.shape[0]
cols_compareallergen = compare_allergen_data.shape[1]
ones_column = np.full((rows_compareallergen, 1), "1")
compare_allergen_data = np.hstack((compare_allergen_data, ones_column))
print(compare_allergen_data.shape)
print(compare_allergen_data)
print("\n\n\n")

#ALLERTOP_ALLERGEN DATA
allertop_allergen_data = readData("allertop_allergen", "wk2_ip_datasets/allergens_allertop.txt")
rows_allertopallergen = allertop_allergen_data.shape[0]
cols_allertopallergen = allertop_allergen_data.shape[1]
ones_column = np.full((rows_allertopallergen, 1), "1")
allertop_allergen_data = np.hstack((allertop_allergen_data, ones_column))
print(allertop_allergen_data.shape)
print(allertop_allergen_data)
print("\n\n\n")

#ALLERTOP_NONALLERGEN DATA
allertop_nonallergen_data = readData("allertop_nonallergen", "wk2_ip_datasets/nonallergens_allertop.txt")
rows_allertopnonallergen = allertop_nonallergen_data.shape[0]
cols_allertopnonallergen = allertop_nonallergen_data.shape[1]
zeros_column = np.full((rows_allertopnonallergen, 1), "0")
allertop_nonallergen_data = np.hstack((allertop_nonallergen_data, zeros_column))
print(allertop_nonallergen_data.shape)
print(allertop_nonallergen_data)
print("\n\n\n")

#UNIPROT_KB ALLERGEN DATA
uniprot_df = pd.read_excel(r"wk2_ip_datasets/uniprotkb_allergen_AND_reviewed_true_2023_08_30.xlsx")
uniprot_df = uniprot_df.values
uniprot_data = []
rows_uniprot = uniprot_df.shape[0]
cols_uniprot = uniprot_df.shape[1]
for i in range(rows_uniprot):
  uniprot_data.append([f"uniprot_allergen{i+1}", uniprot_df[i][cols_uniprot-1]])
uniprot_data = np.array(uniprot_data)
ones_column = np.full((rows_uniprot, 1), "1")
uniprot_data = np.hstack((uniprot_data, ones_column))
print(uniprot_data.shape)
print(uniprot_data)
print("\n\n\n")

#ALLERGEN_ONLINE DATA
df_allergenonline = pd.read_csv("wk2_ip_datasets/allergen_online.csv")
allergenonline_data = df_allergenonline.values

rows_allergenonline = allergenonline_data.shape[0]
cols_allergenonline = allergenonline_data.shape[1]

for i in range(rows_allergenonline):
  allergenonline_data[i][0] = f"allergen_online{i+1}"

ones_column = np.full((rows_allergenonline, 1), "1")
allergenonline_data = np.hstack((allergenonline_data, ones_column))
print(allergenonline_data.shape)
print(allergenonline_data)

"""
STORING ALL THE DATA IN A SINGLE NUMPY ARRAY
The format for the same is as follows:
[name ,  fasta sequence  ,  encoded value]
encoded value is "0" if non allergen and "1" if allergen
"""

all_data = np.vstack((algpred_data, compare_allergen_data, allertop_allergen_data, allertop_nonallergen_data, uniprot_data, allergenonline_data))

print(all_data.shape)
print(all_data)
df=pd.DataFrame(all_data)
df.to_csv('all_data.csv')

"""
Importing all_data with aac features added
The aac features were extracted using the pfeature library locally
"""

df = pd.read_csv("wk2_ip_datasets/all_data_with_aac.csv")
all_data = df.to_numpy()
new_data=[i[1:len(i)] for i in all_data]
print(all_data.shape)
print(new_data)
print(len(new_data))

#Getting rid of duplicates

print(len(new_data))
new_data=np.array(new_data)
tuple_of_tuples = tuple(map(tuple, new_data))#adding data in a set to get rid of the duplicates
s=set(tuple_of_tuples)
print(s)
org_data=[]
for i in s:
  org_data.append(list(i))
org_data=np.array(org_data)#converting the list into numpy array
df=pd.DataFrame(org_data)#converting the numpy array into a dataframe
df.to_csv('real_aac.csv',index=False,header=False)#converting into a csv file

df.shape


cluster_dict = {}

with open("wk2_ip_datasets/1694200452.fas.1.clstr.sorted", 'r') as file:
    cluster_data = file.read().split(">Cluster")

    for cluster in cluster_data:
        lines = cluster.strip().split('\n')
        if not lines:
            continue
        cluster_name = lines[0]  # The first line contains the cluster name
        sequence_count = len(lines) - 1  # Subtract 1 for the cluster name line

        cluster_dict[cluster_name] = sequence_count

# Now, cluster_dict contains cluster names as keys and the number of sequences in each cluster as values
# print(cluster_dict)
sum = 0
del cluster_dict['']
import random

target_sum = 6668
selected_clusters = []

# Create a list of shuffled cluster names (keys)
shuffled_clusters = list(cluster_dict.keys())
random.shuffle(shuffled_clusters)

current_sum = 0

for cluster_name in shuffled_clusters:
    if current_sum + cluster_dict[cluster_name] <= target_sum:
        selected_clusters.append(cluster_name)
        current_sum += cluster_dict[cluster_name]
    if current_sum == target_sum:
        break

# selected_clusters now contains the randomly selected clusters that add up to 6668
print(len(selected_clusters))
#cluster_data has all the sequences in that specific cluster

sequence_ids =[]
for i in selected_clusters:
    lines = (cluster_data[int(i)].split('>'))
    for line in lines[1:]:
        id_part = line.split('...')[0].strip()  # Split by '...' and take the first part
        if id_part:  # Check if the ID part is not empty
            sequence_ids.append(id_part)


validation = []
for i in cluster_dict.keys():
  if i not in selected_clusters:
    validation.append(i)

sequence_ids_2 =[]
for i in validation:
    lines = (cluster_data[int(i)].split('>'))
    for line in lines[1:]:
        id_part = line.split('...')[0].strip()  # Split by '...' and take the first part
        if id_part:  # Check if the ID part is not empty
            sequence_ids_2.append(id_part)

print(validation)


df_positive = pd.read_csv("wk2_ip_datasets/positive.csv")#negative samples aac (non allergens) from the csv file

df_negative = pd.read_csv("wk2_ip_datasets/new_negative.csv")# extracting positive samples aac (allergens) from the csv file

df_positive.columns

plus_set=df_positive.to_numpy()
minus_set=df_negative.to_numpy()
temp_set=np.concatenate((plus_set,minus_set),axis=0)#merging both negative and positive dataset into one dataset
# print(temp_set[0:1])
DATA_set=np.delete(temp_set,[0,1],axis=1)#deleting the first two columns of the 
# print(DATA_set)


new_column_names = {'name': 'IDs', 'sequence': 'Sequence'}
df_negative=df_negative.rename(columns=new_column_names)
Data_set_df=pd.concat([df_positive,df_negative])
Data_set_df=Data_set_df.drop(columns=['IDs','Sequence'])
Data_set_df=Data_set_df.rename(columns={'Unnamed: 22':'Predicted'})


training_positive_df = df_positive[df_positive['IDs'].isin(sequence_ids)]
validation_positive_df = df_positive[df_positive['IDs'].isin(sequence_ids_2)]


print(training_positive_df.shape)
print(validation_positive_df.shape)
# print(validation_positive_df)


cluster_dict = {}

with open("wk2_ip_datasets/1696966524.fas.1.clstr.sorted", 'r') as file:
    cluster_data = file.read().split(">Cluster")

    for cluster in cluster_data:
        lines = cluster.strip().split('\n')
        if not lines:
            continue
        cluster_name = lines[0]  # The first line contains the cluster name
        sequence_count = len(lines) - 1  # Subtract 1 for the cluster name line

        cluster_dict[cluster_name] = sequence_count

# Now, cluster_dict contains cluster names as keys and the number of sequences in each cluster as values
# print(cluster_dict)
sum = 0
del cluster_dict['']
import random

target_sum = 6668
selected_clusters = []

# Create a list of shuffled cluster names (keys)
shuffled_clusters = list(cluster_dict.keys())
random.shuffle(shuffled_clusters)

current_sum = 0

for cluster_name in shuffled_clusters:
    if current_sum + cluster_dict[cluster_name] <= target_sum:
        selected_clusters.append(cluster_name)
        current_sum += cluster_dict[cluster_name]
    if current_sum == target_sum:
        break

# selected_clusters now contains the randomly selected clusters that add up to 6668
print(len(selected_clusters))
#cluster_data has all the sequences in that specific cluster

sequence_ids = []
for i in selected_clusters:
    lines = (cluster_data[int(i)].split('>'))
    for line in lines[1:]:
        id_part = line.split('...')[0].strip()  # Split by '...' and take the first part
        if id_part:  # Check if the ID part is not empty
            sequence_ids.append(id_part)

validation = []
for i in cluster_dict.keys():
  if i not in selected_clusters:
    validation.append(i)

sequence_ids_2 =[]
for i in validation:
    lines = (cluster_data[int(i)].split('>'))
    for line in lines[1:]:
        id_part = line.split('...')[0].strip()  # Split by '...' and take the first part
        if id_part:  # Check if the ID part is not empty
            sequence_ids_2.append(id_part)






training_negative_df = df_negative[df_negative['IDs'].isin(sequence_ids)]
validation_negative_df = df_negative[df_negative['IDs'].isin(sequence_ids_2)]

print(training_negative_df.shape)
print(validation_negative_df.shape)


#Converting the splits into csv files to feed into the pfeature library and extract their AAC features

training_negative_df.to_csv('training_negative_dataset.csv')
validation_negative_df.to_csv('validation_negative_datatset.csv')


#Importing the csv files into a np array with the extracted AAC features

training_negative_df = pd.read_csv("wk2_ip_datasets/training_neg.csv")
validation_negative_df = pd.read_csv("wk2_ip_datasets/valid_neg.csv")

print(training_negative_df.shape)
print(validation_negative_df.shape)

#Converting all the above data to numpy arrays
train_pos = training_positive_df.to_numpy()
val_pos = validation_positive_df.to_numpy()
train_neg = training_negative_df.to_numpy()
val_neg = validation_negative_df.to_numpy()


#Printing their shapes
print("The shape of the postive training dataset is: ", end=" ")
print(train_pos.shape)

print("The shape of the positive validation dataset is: ", end=" ")
print(val_pos.shape)

print("The shape of the negative training dataset is: ", end=" ")
print(train_neg.shape)

print("The shape of the negative validation dataste is: ", end=" ")
print(val_neg.shape)


#Dropping the FASTA sequence column from all the datasets

#Getting rid of serial number column from train_neg and val_neg
train_neg = np.delete(train_neg, 0, 1)
val_neg = np.delete(val_neg, 0, 1)

train_neg = np.delete(train_neg, 1, 1)
val_neg = np.delete(val_neg, 1, 1)

train_pos = np.delete(train_pos, 1, 1)
val_pos = np.delete(val_pos, 1, 1)


print(train_pos.shape)
print(train_neg.shape)


#Making one training dataset

#Getting rid of the protein_name and storing it elsewhere
name_pos = train_pos[:,0]
name_neg = train_neg[:,0]

#Getting rid of the protein_name column from both the train datasets
train_pos = np.delete(train_pos, 0, 1)
train_neg = np.delete(train_neg, 0, 1)

#Converting nan values of the train_neg and val_neg into 0
train_neg[:,-1] = 0
val_neg[:,-1] = 0

#Converting all the data in string format inside the array into float
train_pos = train_pos.astype(float)

final_train = np.vstack((train_pos, train_neg))
print(final_train.shape)

# print("Final Train")
# print(final_train)



































# # ####RandomForestClassifier#######

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data = final_train
X = data[:, :-1]
y = data[:, -1]
print("XXXXX")
print(X)
print(y)
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

x_new_train_all = X_train_all.astype(float)
y_new_train_all = y_train_all.astype(float)

rf_classifier.fit(x_new_train_all, y_new_train_all)
y_pred = rf_classifier.predict(X_test)
print(y_pred.astype(int))
print(y_test.astype(int))

accuracy = accuracy_score(y_test.astype(int), y_pred.astype(int))
print("\nClassification Report:")
print(f"Overall Accuracy: {accuracy:.2%} Using Random Forest")

print(classification_report(y_test.astype(int), y_pred.astype(int)))




# # #######LogisticRegression#########

from sklearn.linear_model import LogisticRegression


data = final_train
X = data[:, :-1]
y = data[:, -1]
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


x_new_train_all = X_train_all.astype(float)
y_new_train_all = y_train_all.astype(float)

print(y_pred.astype(int))
print(y_test.astype(int))

clf = LogisticRegression()

# Train the classifier
clf.fit(x_new_train_all, y_new_train_all)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test.astype(int), y_pred.astype(int))
print("\nClassification Report:")
print(f"Overall Accuracy: {accuracy:.2%} using LogisticRegression Model")


print(classification_report(y_test.astype(int), y_pred.astype(int)))



# # #######Deep Learning#######

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Assuming final_train is your processed data
data = final_train

# Splitting the data into features (X) and labels (y)
X = data[:, :-1]
y = data[:, -1]

# Convert labels to integers
y = y.astype(int)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Calculate and print overall accuracy
accuracy = accuracy_score(y_test.astype(int), y_pred.astype(int))
print(f"Overall Accuracy: {accuracy:.2%}")

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test.astype(int), y_pred.astype(int)))



# # ############EPOCH Enhanced - Hyper-Parameter Tuning Algorithm#############

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Assuming final_train is your processed data
data = final_train

# Splitting the data into features (X) and labels (y)
X = data[:, :-1]
y = data[:, -1]

# Convert labels to integers
y = y.astype(int)

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define hyperparameters to iterate over
epochs_list = [10, 20, 30]
optimizer_list = ['adam', 'sgd']
activation_list = ['relu', 'tanh']

# Iterate over hyperparameters
for epochs in epochs_list:
    for optimizer in optimizer_list:
        for activation in activation_list:
            # Define and compile the model
            model = Sequential([
                Dense(64, activation=activation, input_shape=(X_train_scaled.shape[1],)),
                Dense(32, activation=activation),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
            
            # Train the model
            model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, validation_data=(X_val_scaled, y_val), verbose=0)
            
            # Evaluate the model
            y_pred_prob = model.predict(X_val_scaled)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            accuracy = accuracy_score(y_val, y_pred)
            print(f'Epochs: {epochs}, Optimizer: {optimizer}, Activation: {activation}, Validation Accuracy: {accuracy:.2%} using Hyper-Parameter Tuning Algorithm')


# ####### Sequential Neural Network Algorithm DL #############
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Assuming final_train is your processed data
data = final_train

# Splitting the data into features (X) and labels (y)
X = data[:, :-1]
y = data[:, -1]

# Convert labels to integers
y = y.astype(int)

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define a more complex neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_data=(X_val_scaled, y_val), verbose=1)

# Evaluate the model on the validation set
y_pred_prob = model.predict(X_val_scaled)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Calculate and print the validation accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2%} Using Sequential Neural Network Algorithm')

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))