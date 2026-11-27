import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import  confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load the training dataset
train_data = pd.read_csv('sign_mnist_13bal_train.csv')

# Separate the data (features) and the  classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']   # Target (first column)

# Load the testing dataset
test_data = pd.read_csv('sign_mnist_13bal_test.csv')

# Separate the data (features) and the  classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']   # Target (first column)

# Use this line to get you started on adding a validation dataset
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=30, random_state=0)

# train neural net
neural_net_model = MLPClassifier(hidden_layer_sizes=(32),random_state=42,tol=0.005)

neural_net_model.fit(X_train, y_train)

# Determine model architecture 
layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
layer_size_str = " x ".join(map(str, layer_sizes))
print(f"Training set size: {len(y_train)}")
print()
print(f"Layer sizes: {layer_size_str}")
print()

# predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred_validate = neural_net_model.predict(X_validate)
y_pred = neural_net_model.predict(X_test)

# calculation for validations
total_counts_validate = 0
validate_correct = 0

# Count correct test validations for each class
for true, pred in zip(y_validate, y_pred_validate):
    total_counts_validate += 1
    if true == pred:
        validate_correct += 1

overall_validation_accuracy = (validate_correct / total_counts_validate*100)
print(f"Overall Validation Accuracy: {overall_validation_accuracy:3.1f}%")

#Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Create dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# Count correct test predictions for each class
for true, pred in zip(y_test, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# For comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1

print(f"----------")

# Calculate and print accuracy for each class and overall test accuracy
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] *100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")
overall_accuracy = overall_correct / len(y_test)*100
print(f"Overall Test Accuracy: {overall_accuracy:3.1f}%")
overall_training_accuracy = correct_counts_training / total_counts_training*100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")

# Contain print statements indicating which two or three letters the model most often misidentifies.
misclassified_count = {}

for class_id in total_counts:
    misclassified_count[class_id] = (total_counts[class_id] - correct_counts[class_id])

# Specifically identifying letters 1-10 that the model most often misidentifies
class_names = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    10: "L"}

print(f"----------")

print(f"Most often misclassified letters:")

for class_id in total_counts:
    print(f"{class_names[class_id]}: {total_counts[class_id] - correct_counts[class_id]} incorrect predictions")

print(f"----------")

print(f"What I found - Comparing the two neuron models:")
print()
print(f"The 8 neuron model had a training accuracy of 11.0%, validation accuracy of 13.3%, and test accuracy of 15.4%. (training and validation accuracies within 5%).")
print()
print (f"The 16 neuron model had a training accuracy of 88.0%, validation accuracy of 56.7%, and test accuracy of 60.0%.")
print()
print(f"It performed better on the test, however, its training and validation accuracies had a 31.3% difference.")
