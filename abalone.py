import tensorflow as tf
import numpy as np
import time

data = []

datafile = open("abalonedata.txt","r")
for line in datafile:
	data.append(line)

DATA_SPLIT = 3500

maleAtt = []
femAtt = []
infAtt = []

maleAge = []
femAge = []
infAge = []

for d in data[0:DATA_SPLIT]:
	x = d.split(",")
	atts = x[1:len(x)-1]
	for i in range(len(atts)):
		atts[i]=float(atts[i])
	age = int(x[len(x)-1].strip())
	if(x[0]=="M"):
		maleAtt.append(atts)
		maleAge.append(age)
	if(x[0]=="F"):
		femAtt.append(atts)
		femAge.append(age)
	if(x[0]=="I"):
		infAtt.append(atts)
		infAge.append(age)

tmaleAtt = []
tfemAtt = []
tinfAtt = []

tmaleAge = []
tfemAge = []
tinfAge = []

for d in data[DATA_SPLIT:]:
	x = d.split(",")
	atts = x[1:len(x)-1]
	for i in range(len(atts)):
		atts[i]=float(atts[i])
	age = int(x[len(x)-1].strip())
	if(x[0]=="M"):
		tmaleAtt.append(atts)
		tmaleAge.append(age)
	if(x[0]=="F"):
		tfemAtt.append(atts)
		tfemAge.append(age)
	if(x[0]=="I"):
		tinfAtt.append(atts)
		tinfAge.append(age)

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



