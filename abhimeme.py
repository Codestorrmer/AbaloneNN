import tensorflow as tf
import numpy as np
import time


def net_run():
    fcs = [tf.contrib.layers.real_valued_column("", dimension = 8)]

    classifier = tf.contrib.learn.DNNClassifier(feature_columns = fcs, hidden_units[30], n_classes = 6, model_dir="/tmp/abalone_model")

    #training_set is defined by the inputs

    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)
        return x, y

    classifier.fit(input_fn=get_train_inputs, steps=2000)

    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)

        return x, y

    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]

    print("\nAccuracy: {0:f}\n".format(accuracy_score))
