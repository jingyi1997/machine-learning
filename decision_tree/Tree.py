
# coding: utf-8

# In[2]:


import numpy as np
import csv
with open('Watermelon-train1.csv',encoding='gbk') as train_f:
        train_data=list(csv.reader(train_f))


# In[3]:


train_data = np.array(train_data)
train_data


# In[4]:


#global variable
feature_num = train_data.shape[1]-2


# In[5]:


def sub_infogain(labels):
    unique_labels = np.unique(labels)
    res = 0
    total_num= labels.shape[0]
    for unique_label in unique_labels:
        sub_num =np.sum((labels==unique_label))
        sub_prob= sub_num/total_num
        inc = sub_prob*np.log2(sub_prob)
        res+=inc
    return -res

    


# In[6]:


#returns the index of the feature which brings the maximum
#information gain
def find_feature(dataset,labels):
    max_fea = -1
    #first calculate the entropy of the information without any feature
    base_info = sub_infogain(labels)
    #the initial calue also means the threshold
    max_gain =0
    for i in range(feature_num):
        #if the feature has been chosen,continue
        curr_info= 0
        if i in tree_fea:
            continue
        else:
            vals = dataset[:,i]
            uni_vals = np.unique(vals)
            #for each unique feature value, accumulate the information gain
            for val in uni_vals:
                sub_index = np.where(vals==val)
                sub_set= dataset[sub_index,:]
                sub_labels = labels[sub_index]
                sub_num = len(sub_index[0])
                prob = sub_num/vals.shape[0]       
                #the function 'sub_infogain' returns the base entropy of information of the subset
                curr_info = prob*sub_infogain(sub_labels)
            if base_info-curr_info>max_gain:
                max_fea = i
                max_gain = curr_info-base_info
    return max_fea
            


# In[7]:


#define a recursive function to build the decision tree, each call for the 
#function should return either a number, which 
#represents the label of the data with the feature, or a dictionary, whose 
#key means the unique value of the feature within 
#the dataset and the value is another dictionary
def decision_tree(dataset,labels):
    #first check if the labels in the dataset are the same
    if np.unique(labels).shape[0]==1:
        #return the label
        return labels[0]
    else:
        #find_feature returns the index of the feature, which starts from zerox
        maxinfo_fea = find_feature(dataset,labels)
        if maxinfo_fea<0:
            #if the information gain is too little or there are no features left to split, return the label 
            #which appeared most
            num= 0
            most_label="1"
            for lab in set(labels):
                curr_num= np.sum(labels==lab)
                if num<curr_num:
                    num=curr_num
                    most_label = lab
            return most_label
        #otherwise split the dataset with according to the selected feature and build the sub tree
        else:                       
            #append the feature to the global variable 
            tree_fea.append(maxinfo_fea)
            #recursive build the decision tree
            #initialize a dictionary first
            child_tree={}
            #extract the values of the dataset on the feature
            vals = dataset[:,maxinfo_fea]
            #print (vals)
            uni_vals = np.unique(vals)
            #for each unique value of the feature, rebuild the dataset and the labels to 
            #build the decision tree recursively
            child_tree[0]=maxinfo_fea
            for val in uni_vals:
                ##'where' returns the indiceof an array which satisfy the condition
                sub_index = (np.where(vals==val))[0]
                sub_set= dataset[sub_index,:]
                sub_labels = labels[sub_index]
                child_tree[val]= decision_tree(sub_set,sub_labels)
            #remove the feature from the global variable so that it can be used for 
            #other trees
            tree_fea.remove(maxinfo_fea)
            return child_tree
          
            


# In[8]:


my_dataset = train_data[1:,1:feature_num+1]


# In[9]:


my_labels = train_data[1:,-1]


# In[10]:


my_tree={}
tree_fea=[]
my_tree = decision_tree(my_dataset,my_labels)
print ((my_tree))


# In[ ]:



    
        


# In[11]:


print(tree_fea)
with open('Watermelon-test1.csv',encoding='gbk') as test_f:
        test_data=list(csv.reader(test_f))


# In[12]:


test_data = np.array(test_data)
test_dataset = test_data[1:,1:feature_num+1]
test_labels = test_data[1:,-1]
test_data


# In[13]:



def test_tree(test_dataset,test_labels):
    corr_num=0
    for index,data in enumerate(test_dataset):
        print (data,end="")
        curr_tree= my_tree
        while(1):
            i= curr_tree[0]
            my_fea= data[i]
            curr_tree = curr_tree[my_fea]
            if curr_tree =="是"or curr_tree=="否":
                
                ans= curr_tree
                fact= test_labels[index]
                print(":   预测值为 "+curr_tree+" , 真实值为 "+fact)
                if ans==fact:
                    corr_num+=1
                break
    print("准确率为"+str( corr_num/test_dataset.shape[0]))
        
            
        


# In[14]:


test_tree(test_dataset,test_labels)

