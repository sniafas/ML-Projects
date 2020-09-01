'''
This implementation is an example of a fcnn in native tensorflow api
Insired from stanford's course CS20: Tensorflow for Deep Learning Research
'''
#!/usr/bin/env python
# coding: utf-8

# System libraries
from time import time
import os

# Helper libraries
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import colors

# Custom libraries
# import dl_utils as utils
import datasets

# Enable tf v1 behavior as in v2 a lot have changed
tf.disable_v2_behavior()
print("TF v1 behaviour enabled")
print("Tensorflow version", tf.__version__)

class NeuralNet:
    '''
    Build a NN in tf
    '''
    def __init__(self, n_x=1000, noise_x=0, session_type='cpu'):
        self.learning_rate = 0.001
        self.verbose = 0
        self.batch_size = 64
        self.n_x = n_x
        self.noise_x = noise_x
        tf.reset_default_graph()
        if session_type == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.sess = tf.Session()
        else:
            self.sess = utils.gpu_session()

    def get_data(self):
        '''
        Fetch and create dataset
        '''
        # Fetch dataset
        X, Y = datasets.data_spiral(self.n_x, self.noise_x)

        # Fetch test set - grid points
        X_test, y_test, _, _ = datasets.grid_points()

        # Split dataset for train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.3)
        self.valid_len = len(X_valid)
        print("Train size", X_train.shape)
        print("Valid size", X_valid.shape)
        print("Test size", X_test.shape)

        try:
            assert X_train.shape[1] == X_test.shape[1]
        except AssertionError as e:
            raise AssertionError("Dataset input size is not equal. %s"%e)

        with tf.name_scope('data'):

            # Define placeholders for test set, since eval() is used
            x_test_pl = tf.placeholder(dtype=X_test.dtype, shape=X_test.shape)
            y_test_pl = tf.placeholder(dtype=y_test.dtype, shape=y_test.shape)

            # Create tensorflow compatible dataset with tf.data api (no placeholders)
            train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(
                self.batch_size,
                drop_remainder=True)
            valid_data = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(
                self.batch_size,
                drop_remainder=False)
            test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
                self.batch_size,
                drop_remainder=False)

            ## Define Iterators
            self.iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                            train_data.output_shapes)

            self.train_init_op = self.iterator.make_initializer(train_data)
            self.valid_init_op = self.iterator.make_initializer(valid_data)
            self.test_init_op = self.iterator.make_initializer(test_data)

            # Create test feed dict for inference
            self.test_feed_dict = {x_test_pl: X_test,
                                   y_test_pl: y_test}

            # Take the next sample-label for the infered set into the computational Graph
            self.sample, self.labels = self.iterator.get_next()
            #self.sample = tf.reshape(self.sample, [-1,1])
            self.labels = tf.one_hot(self.labels, depth=2, name='y1h')
            #self.labels = tf.reshape(self.labels, [-1,1])
            self.labels = tf.cast(self.labels, tf.float32)
            self.sample = tf.cast(self.sample, tf.float32)

        return X_train, X_valid, y_train, y_valid

    def model(self):
        '''
        Build model architecture using functional API
        '''
        with tf.name_scope('architecture'):
            l1 = tf.layers.Dense(8,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.keras.initializers.glorot_normal(),
                                 name="hidden_1")(self.sample)
            l2 = tf.layers.Dense(8,
                                 activation=tf.nn.relu,
                                 name="hidden_2")(l1)
            l3 = tf.layers.Dense(8,
                                 activation=tf.nn.relu,
                                 name="hidden_2")(l2)
            l4 = tf.layers.Dense(8,
                                 activation=tf.nn.relu,
                                 name="hidden_2")(l3)

            self.logits = tf.layers.Dense(2,
                                          activation=tf.nn.softmax,
                                          name="output")(l4)
            #self.logits = tf.transpose(self.logits)

    def loss_function(self):
        '''
        Compute loss of the model
        '''
        with tf.name_scope('loss'):

            #entropy = tf.keras.losses.binary_crossentropy(self.labels,self.logits,
            #                                              from_logits=True)
            #entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,
            #                                                  logits=self.logits)
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                              logits=self.logits)
            self.loss = tf.reduce_mean(entropy, axis=0, name='loss')

        print("Labels shape:", self.labels.shape)
        print("Logits shape:", self.logits.shape)

    def optimize(self):
        '''
        Define Optimizer method to minimize loss
        '''
        with tf.name_scope('optimize'):
            self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            #self.opt = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def evaluate(self):
        '''
        Compute predictions and accuracy in a batch
        '''
        with tf.name_scope('predict'):

            self.predictions = tf.nn.sigmoid(self.logits)
            self.correct_preds = tf.equal(tf.argmax(self.predictions, -1),
                                          tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(self.correct_preds, tf.float32))

    def build(self):
        '''
        Build the computation graph
        '''
        self.model()
        self.loss_function()
        self.optimize()
        self.evaluate()

    def train_predict(self, epochs):
        '''
        Train and evaluate model
        :param epochs: number of epochs
        :return: predictions
        '''
        self.sess.run(tf.global_variables_initializer())
        print("Start training process...")
        for epoch in range(epochs):
            self.sess.run(self.train_init_op)
            total_loss = 0
            n_batches = 0
            sum_train_acc = 0
            try:
                while True:
                    _, loss, train_acc = self.sess.run([self.opt, self.loss, self.accuracy])
                    total_loss += loss
                    sum_train_acc += train_acc
                    n_batches += self.batch_size

            except tf.errors.OutOfRangeError:
                pass

            if epoch % 20 == 0:
                print("Epoch: {}, Average loss: {:.4f}, Train Accuracy: {:.2f}%".format(
                    epoch,
                    total_loss/n_batches,
                    (sum_train_acc/n_batches)*100))

        ## Evaluation ##
        print("\nEvaluating...")
        self.sess.run(self.valid_init_op)
        sum_acc = 0
        sum_loss = 0
        sum_preds = []
        try:
            while True:
                batch_val_loss, batch_val_acc, preds = self.sess.run([self.loss,
                                                                      self.accuracy,
                                                                      self.predictions])
                sum_loss += batch_val_loss
                sum_acc += batch_val_acc
                sum_preds.append(preds)

        except tf.errors.OutOfRangeError:
            pass
        print("Average Validation loss: {:.4f}, Validation Accuracy: {:.2f}%".format(
            sum_loss/self.valid_len,
            (sum_acc/self.valid_len)*100))

        return np.concatenate(sum_preds)

    def test_plot(self, predictions, x_valid):
        '''
        Infer test set to the trained model and plot predictions
        :param predictions: array of predicted labels
        :return: None
        '''
        ## Test Set ##
        self.sess.run(self.test_init_op)
        z_preds = []
        try:
            while True:
                z = self.predictions.eval(session=self.sess, feed_dict=self.test_feed_dict)
                z_preds.append(z)
        except tf.errors.OutOfRangeError:
            pass

        plt.figure(figsize=(5, 5), dpi=75)
        colormap = colors.ListedColormap(["#f59322", "#e8eaeb", "#0877bd"])
        _, _, xx, yy = datasets.grid_points()

        z_preds = np.concatenate(z_preds)
        z_preds = np.argmax(z_preds, axis=1)

        plt.scatter(x_valid[:, 0], x_valid[:, 1],
                    c=predictions[:, 1], edgecolors='k', s=50, cmap=colormap)
        plt.contourf(xx, yy, z_preds.reshape(xx.shape), cmap=colormap, alpha=0.4)
        plt.show()

if __name__ == '__main__':

    nn = NeuralNet(n_x=10000, noise_x=30)
    x_train, x_valid, y_train, y_valid = nn.get_data()
    nn.build()
    model_train = time()
    preds = nn.train_predict(30)
    print('Model train & evaluation time: %.1f sec' % (time() - model_train))
    nn.test_plot(preds,x_valid)
