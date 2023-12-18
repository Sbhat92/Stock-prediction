import pickle

# Load the history from the file
with open('history_model_encoded.pkl', 'rb') as file:
    history = pickle.load(file)

# Extract the accuracy values
accuracy_values = history['hinge_acc']  # Change 'accuracy' to 'val_accuracy' if it's validation accuracy

# Find the highest accuracy
highest_accuracy = (accuracy_values)[-1]

print(f"The highest accuracy is: {highest_accuracy}")
