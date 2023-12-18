import pickle
import matplotlib.pyplot as plt
# Load the history from the file
with open('history_model_encoded.pkl', 'rb') as file:
    history = pickle.load(file)

# Extract the accuracy values
accuracy_values = history['hinge_acc']  # Change 'accuracy' to 'val_accuracy' if it's validation accuracy

# Find the highest accuracy
highest_accuracy = accuracy_values[-1]

print(f"The highest accuracy is: {highest_accuracy}")

# Plot training history
plt.figure(figsize=(12, 6))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history['hinge_acc'])
plt.plot(history['val_hinge_acc'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])


plt.tight_layout()
plt.show()
