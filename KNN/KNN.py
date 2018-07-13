# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np

# <codecell>

import csv

# <codecell>

import matplotlib as mpl
import matplotlib.pyplot as plt

# <codecell>

import operator

# <markdowncell>

# load the train and rest data from the file

# <codecell>

train_file='semeion_train.csv'
test_file='semeion_test.csv'
with open(train_file) as train_f:
    train_data=list(csv.reader(train_f))
with open(test_file) as test_f:
    test_data=list(csv.reader(test_f))            

# <markdowncell>

# Define a function to change the CSV data into list

# <codecell>

def createData(data,datasets,labels):
    for line in data:
        line=line[0]
        datas= line.split(' ')
        datas_num=len(datas)
        float_datas=[]
        label_datas=[]
        for i in range(datas_num-11):
            float_datas.append(float(datas[i]))
        datasets.append(float_datas)
        for i in range(datas_num-11,datas_num-1):
            label_datas.append(int(datas[i]))
        labels.append(label_datas)

# <markdowncell>

# Define the classifier function using kNN 

# <codecell>

def kNN_classifier(inX, dataset, labels, k):
    sample_num=dataset.shape[0]
    diff= np.tile(inX,(sample_num,1))-dataset
    sqr_mat= diff**2
    sqr_dis= sqr_mat.sum(axis=1)
    '''dis is a vector which stores the distance between the inX and each of the training set'''
    dis=sqr_dis**0.5
    '''argsort assendingly sorts the distance and  return the corresponding array of indice'''
    sorted_index=dis.argsort()
    results=np.array(np.zeros(10))
    for i in range(k):
        '''class_i is a vector which stands for the label of the i-th nearest vector'''
        class_i= labels[sorted_index[i]]
        results+=class_i
    '''the biggest element indicates the corresponding index appears most times'''
    '''print results'''
    res= results.argmax()
    return res

# <markdowncell>

# Initialize the training and testing data

# <codecell>

train_data=np.array( train_data);
test_data=np.array( test_data);
train_labels=[];
train_datasets=[];
test_labels=[];
test_datasets=[];
createData(train_data,train_datasets,train_labels);
createData(test_data,test_datasets,test_labels);
test_datasets=np.array(test_datasets);
train_datasets=np.array(train_datasets);
test_labels=np.array(test_labels);
train_labels=np.array(train_labels);

# <codecell>

for k in [1,3,5,10]:
    right_num=0;
    for i in range(test_datasets.shape[0]):
        '''print kNN_classifier(test_datasets[i],train_datasets,train_labels,1),test_labels[i].argmax()'''
        if kNN_classifier(test_datasets[i],train_datasets,train_labels,k)==test_labels[i].argmax():
            right_num+=1
    '''print right_num'''
    '''print test_datasets.shape[0]'''
    print "The classification error ratio for k= %d is %.4f"%(k, round(1-float(right_num)/float(test_datasets.shape[0]),4))

# <markdowncell>

# Divide a cross-validation set from the training set and determine the best K

# <codecell>

'''the number of element in the cross validation set'''
all_num= train_datasets.shape[0]
cv_num=int(all_num*0.2)
'''print type(cv_num)'''
get_index=range(0,all_num);
'''generate the index of element chosen for the cross validation set'''
np.random.shuffle(get_index);
'''print get_index'''
train_index= get_index[0:cv_num]
val_index=get_index[cv_num:all_num]
'''print train_index'''
'''print val_index'''
'''extract the training data for the cross validation'''
cv_train=train_datasets[train_index,:]
cv_train_labels=train_labels[train_index,:]
'''extract the validation data for the cross validation'''
cv_val=train_datasets[val_index,:]
cv_val_labels=train_labels[val_index,:]
cv_res = []
for k in range(1,11):
    right_num= 0
    temp_res=0
    for i in range(len(cv_val)):
        if kNN_classifier(cv_val[i],cv_train,cv_train_labels,k)==cv_val_labels[i].argmax():
            right_num+=1
    temp_res = round(1-float(right_num)/float(len(cv_val)),4)
    cv_res.append(temp_res)
    print "The classification error ratio for k= %d is %.4f"%(k,temp_res )

# <codecell>

plt.figure('cv_figure')
plt.ylim(0,1)
plt.xlabel('k Value')
plt.ylabel('error rate')
for i in range(1,11):
   ann_str= '['+str(i)+','+str(cv_res[i-1])+']'
   plt.annotate(ann_str,xy=(i,cv_res[i-1]),xytext=(i,cv_res[i-1]))
plt.plot(range(1,11),cv_res,'r')
plt.scatter(range(1,11),cv_res)
plt.savefig('result.png')
plt.show()

# <codecell>


