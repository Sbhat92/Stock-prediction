import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import datetime
from sklearn.preprocessing import robust_scale
from tqdm.auto import tqdm
import argparse
import matplotlib.pyplot as plt
from preprocessing import *
from AdvALSTM import *
import pickle

class TestEvaluation(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        super(TestEvaluation, self).__init__()
        self.X_test, self.y_test = X_test, y_test
        
    def on_epoch_end(self, epoch, logs):
        X_test, y_test = self.X_test, self.y_test
        test_metrics = self.model.evaluate(self.X_test, self.y_test, verbose = 0)

        logs["test_loss"] = test_metrics[0]
        logs["test_acc"] = test_metrics[1]


def main():
    # Load preprocessed data
    with open('preprocessed_data.npz', 'rb') as f:
        X_train = np.load(f, allow_pickle=True)
        y_train = np.load(f, allow_pickle=True)
        X_validation = np.load(f, allow_pickle=True)
        y_validation = np.load(f, allow_pickle=True)
        X_test = np.load(f, allow_pickle=True)
        y_test = np.load(f, allow_pickle=True)

    # Create model
    model = AdvALSTM(
        units = 15, 
        epsilon = 0.1, 
        beta = 0.05,    
        learning_rate = 1E-2, 
        l2 = 0.001, 
        attention = True, 
        hinge = True,
        dropout = 0.0,
        adversarial_training = True,
        random_perturbations = False
    )

    history = model.fit(
        X_train,  y_train,
        validation_data = (X_validation, y_validation),
        epochs = 100,
        batch_size = 1024,
        callbacks=[TestEvaluation(X_test, y_test)]
    )
    with open('history_model_encoded.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    # Plot training history
    plt.figure(figsize=(12, 6))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['hinge_acc'])
    plt.plot(history.history['val_hinge_acc'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])


    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()