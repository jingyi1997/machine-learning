
# coding: utf-8

# In[32]:


import numpy as np
import csv
with open('Watermelon-train2.csv',encoding='gbk') as train_f:
        train_data=list(csv.reader(train_f))


# In[33]:


train_data = np.array(train_data)


# In[ ]:





# In[35]:


feature_num= train_data.shape[1]-2
dis_fea=np.array([0,1,2,3])
my_traindata= train_data[1:,1:feature_num+1]
my_labels= train_data[1:,-1]


# In[36]:


def CalGini(labels):
    unique_labels = np.unique(labels)
    res = 0
    total_num= labels.shape[0]
    for unique_label in unique_labels:
        sub_num =np.sum((labels==unique_label))
        sub_prob= sub_num/total_num
        inc = sub_prob*sub_prob
        res+=inc
    return 1-res


# In[37]:


def findBestFeature(dataset,labels):
    fea_name=-1
    fea_val=0
    sub_set1=np.array([])
    sub_label1=np.array([])
    sub_set2=np.array([])
    sub_label2=np.array([])
    best_gini =CalGini(labels)
    for i in range(feature_num):
        vals = dataset[:,i]
        uni_val= np.unique(vals)
        if uni_val.shape[0]==1:
            continue
        else:
            #if the feature is discrete
            if i in dis_fea:
                uni_val2=uni_val
            else:
                uni_val2=[]
                uni_val=uni_val.astype(np.float64)
                #calculate n-1 partition point for continuous feature
                for index,con_val in enumerate(uni_val):
                    if index==uni_val.shape[0]-1:
                        continue
                    else:
                        var1 = uni_val[index+1]
                        var2 = (con_val)                        
                        uni_val2.append((var1+var2)/2)
            for val in uni_val2:
                #for discrete the feature, choose the sample with the feature value
                if i in dis_fea:
                    sub_index1 = (np.where(vals==val))[0]
                else:
                #for continuous feature, choose the sample with the feature value below the partition point
                    vals=vals.astype(np.float64)
                    sub_index1 = (np.where(vals<=val))[0]
                sub_set1_temp= dataset[sub_index1,:]
                sub_label1_temp= labels[sub_index1]
                sub_index2= np.array(list(set(range(dataset.shape[0]))-set(sub_index1)))
                sub_set2_temp= dataset[sub_index2,:]
                sub_label2_temp = labels[sub_index2]
                temp_gini=len(sub_index1)/dataset.shape[0]*CalGini(sub_label1_temp)+len(sub_index2)/dataset.shape[0]*CalGini(sub_label2_temp)
                #when the temp gini index is the minimum, update the gini index
                if temp_gini<best_gini:
                    fea_name=i
                    fea_val=val
                    best_gini=temp_gini
                    sub_set1=sub_set1_temp
                    sub_set2=sub_set2_temp
                    sub_label1=sub_label1_temp
                    sub_label2=sub_label2_temp                   
    return fea_name,fea_val,sub_set1,sub_set2,sub_label1,sub_label2


# In[38]:


findBestFeature(my_traindata,my_labels)


# In[39]:



def cart_tree(dataset,labels):
     if np.unique(labels).shape[0]==1:
        #if all samples in the dataset fall into one label,return the label directly
        return labels[0]
     else:
        #'findBestFeature’ return the feature，the partition point and the divided datasets
        #with the minimum Gini index.
        #If the Gini index's decrease is under a certain threshold, which means there is no 
        #need to split the dataset, return the label which appeared most
        fea_name,fea_val,sub_set1,sub_set2,sub_label1,sub_label2 = findBestFeature(dataset,labels)
        if fea_name==-1:
            num= 0
            most_label="1"
            for lab in set(labels):
                curr_num= np.sum(labels==lab)
                if num<curr_num:
                    num=curr_num
                    most_label = lab
            return most_label
        else:
            #otherwise call itself reursively to build the subtree
            res={}
            res['name']=fea_name
            res['val']=fea_val
            res[0]=cart_tree(sub_set1,sub_label1)
            res[1]=cart_tree(sub_set2,sub_label2)
            return res
             
                
            


# In[40]:


mytree= cart_tree(my_traindata,my_labels)


# In[41]:


print(mytree)


# In[43]:


with open('Watermelon-test2.csv',encoding='gbk') as test_f:
        test_data=list(csv.reader(test_f))
test_data = np.array(test_data)
test_dataset = test_data[1:,1:feature_num+1]
test_labels = test_data[1:,-1]


# In[44]:


def test_tree(test_dataset,test_labels):
    corr_num=0
    for index,data in enumerate(test_dataset):
        curr_tree= mytree
        print (data,end="")
        while(1):
            if curr_tree =="是"or curr_tree=="否":
                    ans= curr_tree
                    fact= test_labels[index]
                    if ans==fact:
                        corr_num+=1
                    print(":   预测值为 "+curr_tree+" , 真实值为 "+fact)
                    break
            else:
                fea_num=curr_tree['name']
                fea_val=curr_tree['val']
                fea_index=1
                if fea_num in dis_fea:
                    if data[fea_num]==fea_val:
                        fea_index=0
                else:
                    temp=float(data[fea_num])
                    if temp<=fea_val:
                        fea_index=0
                curr_tree=curr_tree[fea_index]
    return corr_num/test_dataset.shape[0]


# In[45]:


test_tree(test_dataset,test_labels)


# In[ ]:




