# System libraries
from time import time
import random

# Helper libraries
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import colors

# Custom libraries
# import dl_utils as utils
import datasets
    
print("Tensorflow version",tf.__version__)

class NeuralNet:

    def __init__(self, n_X = 1000, noise_X = 0, session_type='cpu'):

        self.lr = 0.001
        self.n_X = n_X
        self.noise_X = noise_X

    def get_data(self):

        X, Y = datasets.data_spiral(self.n_X, self.noise_X)
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)
        
        # Create a grid of points to predict
        x, _, xx, yy = datasets.grid_points()

        return X_train, X_test, y_train, y_test, x, xx, yy

    def create_model(self):

        model = tf.keras.Sequential()

        # Input Layer
        model.add(tf.keras.layers.Dense(8, input_dim=2, activation='relu'))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(8, activation='relu'))

        # Output Layer
        model.add(tf.keras.layers.Dense(2,activation='softmax'))

        # Compile model
        model.compile(loss='sparse_categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(self.lr),
                     metrics=['accuracy'])

        return model

    def train_predict(self, X_train, X_test, y_train, y_test, x, epochs):

        model_train = time()
        results = model.fit(X_train,y_train,
                            batch_size=64,
                            epochs=epochs,
                            validation_data=(X_test,y_test),
                            verbose=2
                           )
        # Test predictions
        predictions = model.predict(X_test)
        keras_train = time() - model_train
        print('Model train & evaluation time %.1f sec' % keras_train)

        # Grid predictions
        z_preds = model.predict(x)

        print("Evaluating on train set...")
        (loss, accuracy) = model.evaluate(X_train, y_train.T, verbose=0)
        print("loss={:.4f}, accuracy: {:.2f}%".format(loss,accuracy * 100))

        print("Evaluating on test set...")
        (loss, accuracy) = model.evaluate(X_test, y_test.T, verbose=0)
        print("loss={:.4f}, accuracy: {:.2f}%".format(loss,accuracy * 100))

        return z_preds, predictions

    def grid_plot(self, z_preds, predictions):

        plt.figure(figsize=(5, 5), dpi=75)
        colormap = colors.ListedColormap(["#f59322", "#e8eaeb", "#0877bd"])
        z_preds = np.argmax(z_preds,axis=1)
        plt.scatter(X_test[:,0],X_test[:,1], c=predictions[:,1], edgecolors='k', s=50, cmap=colormap)
        plt.contourf(xx,yy,z_preds.reshape(xx.shape), cmap=colormap, alpha=0.4)
        plt.show()

if __name__ == '__main__':

    nn = NeuralNet(n_X = 10000, noise_X = 30)
    X_train, X_test, y_train, y_test, x, xx, yy = nn.get_data()
    model = nn.create_model()
    z_preds, predictions = nn.train_predict(X_train, X_test, y_train, y_test, x, 30)
    nn.grid_plot(z_preds,predictions)
    