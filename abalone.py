import tensorflow as tf
import numpy as np

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




