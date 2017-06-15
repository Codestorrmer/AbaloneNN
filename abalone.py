import tensorflow as tf
import numpy as np
import time

data = []

datafile = open("abalonedata.txt","r")
for line in datafile:
	data.append(line)

DATA_SPLIT = 3500
MINI_BATCH_SIZE = 10
EPOCHS = 150
CHECK = 1
NUM_HIDDEN = 50
NUM_HIDDEN_2 = 100
DROPOUT = 0.5
PRINTA = 10
LEARNING_RATE = 5e-3
#true->print test accuracy, false-> print training accuracy
PRINTW = False


def makeOutput(arr):
	ret = []
	for i in range(len(arr)):
		a = []
		for x in range(6):a.append(0)
		a[arr[i]]=1
		ret.append(a)
	return ret

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
	age = int(x[len(x)-1].strip())/5
	if(x[0]=="M"):
		maleAtt.append(atts)
		maleAge.append(age)
	if(x[0]=="F"):
		femAtt.append(atts)
		femAge.append(age)
	if(x[0]=="I"):
		infAtt.append(atts)
		infAge.append(age)

maleAge = makeOutput(maleAge)
femAge = makeOutput(femAge)
infAge = makeOutput(infAge)


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
	age = int(x[len(x)-1].strip())/5
	if(x[0]=="M"):
		tmaleAtt.append(atts)
		tmaleAge.append(age)
	if(x[0]=="F"):
		tfemAtt.append(atts)
		tfemAge.append(age)
	if(x[0]=="I"):
		tinfAtt.append(atts)
		tinfAge.append(age)

tmaleAge = makeOutput(tmaleAge)
tfemAge = makeOutput(tfemAge)
tinfAge = makeOutput(tinfAge)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None,7])
y_ = tf.placeholder(tf.float32, shape = [None,6])


W_0 = weight_variable([7,NUM_HIDDEN])
b_0 = bias_variable([NUM_HIDDEN])
h_0 = tf.nn.relu(tf.matmul(x,W_0)+b_0)

keep_prob = tf.placeholder(tf.float32)
h_0_drop = tf.nn.dropout(h_0, keep_prob)

W_1 = weight_variable([NUM_HIDDEN,NUM_HIDDEN_2])
b_1 = bias_variable([NUM_HIDDEN_2])
h_1 = tf.nn.relu(tf.matmul(h_0,W_1)+b_1)

W_2 = weight_variable([NUM_HIDDEN_2,6])
b_2 = bias_variable([6])
y_2 = tf.matmul(h_1,W_2)+b_2

#Use to toggle between M/F/I
trainAtt = infAtt
testAtt = tinfAtt
trainAge = infAge
testAge = tinfAge

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_2))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_2,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
#sess.run(tf.global_variables_initializer())
saver.restore(sess, "abaloneNet")
for i in range(1,EPOCHS+1):
	for j in range(len(trainAtt)/MINI_BATCH_SIZE):
		xdata = trainAtt[j*MINI_BATCH_SIZE:(j+1)*MINI_BATCH_SIZE]
		ydata = trainAge[j*MINI_BATCH_SIZE:(j+1)*MINI_BATCH_SIZE]
		train_step.run(feed_dict={x: xdata, y_: ydata, keep_prob: DROPOUT})	
	if(len(trainAtt)%MINI_BATCH_SIZE != 0):
		s = (len(trainAtt)/MINI_BATCH_SIZE)*MINI_BATCH_SIZE
		xdata = trainAtt[s:]	
		ydata = trainAge[s:]
		train_step.run(feed_dict={x: xdata, y_: ydata, keep_prob: DROPOUT})	
	if(i%PRINTA==0):
		print("Epoch %d"%(i))
		if(PRINTW):
			print("test accuracy %g"%accuracy.eval(feed_dict={x: testAtt, y_: testAge, keep_prob: 1.0}))
		else:
			print("training accuracy %g"%accuracy.eval(feed_dict={x: trainAtt, y_: trainAge, keep_prob: 1.0}))

save_path = saver.save(sess, "abaloneNet")


