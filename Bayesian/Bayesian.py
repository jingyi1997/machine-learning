
# coding: utf-8

# In[7]:


import csv
import numpy as np
import math
def createData(filename):
    with open(filename) as train_f:
        train_data=list(csv.reader(train_f))
    #devide the train_data by classes
    separated ={}
    for i in range(1,len(train_data)):
        vector = train_data[i]
        # The label for the sample
        class_num=int(vector[-1])
        #The first sample in a class
        if (class_num not in separated):
            separated[class_num] = []
        separated[class_num].append(vector)
    #for each class, create data on each demension
    separated_data={}
    for class_num in separated:
        separated_data[class_num]={}
        for demen_num,demen_name in enumerate(train_data[0]):
            #print (type(demen_name))
            if demen_name!='Label' and demen_name!='No.':
                separated_data[class_num][demen_name]=np.array(separated[class_num])[:,demen_num]
    return len(train_data),separated_data
        
    
        
        
    
    


# In[8]:


total_trainnum,train_data=createData('train.csv')
for class_num in train_data.keys():
    print("The data in class %d are :"%(class_num),end='\n')
    for demen_name in train_data[class_num].keys():
        print ("   ",demen_name,train_data[class_num][demen_name], end='\n')
        
    






# In[9]:


#Preprocess the sample data and extract some features of the data
def pro_data(train_data):
    pro_data={}
    for class_num in train_data.keys():
        pro_data[class_num]={}
        for demen_name in train_data[class_num].keys():
            #print (demen_name)
            pro_data[class_num][demen_name]={}
            #load the raw data 
            sample_data= train_data[class_num][demen_name]
            #print (type(sample_data))
            temp_data=[]
            
            #print (temp_data)
            # for continuous data  calculate the mean value and standard deviation
            if demen_name=='Density' or demen_name=='SugerRatio':
                #convert the type from string to float 
                temp_data=sample_data.astype('float')
                #calculate the mean value
                mean_val=np.mean(np.array(temp_data))
                pro_data[class_num][demen_name]['mean']=mean_val
                std_val=np.std(np.array(temp_data))
                #calculate the standard deviation
                pro_data[class_num][demen_name]['std']=std_val
                pro_data[class_num][demen_name]['sample']=temp_data  
            else:
                temp_data=sample_data.astype('int')
                pro_data[class_num][demen_name]['sample']=temp_data       
    return pro_data
                
                
                


# In[10]:




# In[11]:


train_dataset=pro_data(train_data)
#for class_num in train_dataset.keys():
    #print("The feature of the data in class %d are :"%(class_num),end='\n')
    #for demen_name in train_dataset[class_num].keys():
        #print ("  ",demen_name,':')
        #for feature in train_dataset[class_num][demen_name].keys():
            #print("               ",feature, train_dataset[class_num][demen_name][feature])


# In[12]:


def cal_prob(file_name,train_data):
    with open(file_name) as test_f:
        test_data=list(csv.reader(test_f))
    total_corr= 0
    print (test_data)
    for i in range(1,len(test_data)):
        class_prob={}
        pred_class=0
        pred_prob=0
        #Calculate the possibility for each class
        for class_num in train_data.keys():
            res=1
            
            #Calculate the likelihood on each demension
            for demen_num,demen_name in enumerate(test_data[0]):
                if demen_name!='Label' and demen_name!='No.':                    
                    x = float(test_data[i][demen_num])
                    #When the data follows continuous distribution
                    if demen_name=='Density' or demen_name=='SugerRatio' :
                        #Find the data in training dataset for the class on this demension
                        mean= train_data[class_num][demen_name]['mean']
                        std=train_data[class_num][demen_name]['std']
                        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
                        prob = (1 / (math.sqrt(2*math.pi) * std)) * exponent
                        res=res*prob
                    #When the data follows discrete distribution
                    else:
                        x = int(test_data[i][demen_num])
                        sample_data = train_data[class_num][demen_name]['sample']
                        is_equal=(sample_data==x)
                        prob = sum(is_equal.astype(int))/len(sample_data)
                        res = res * prob                        
            temp_prob=res*(len(train_dataset[class_num])/total_trainnum)
            class_prob[class_num]=temp_prob
            if(temp_prob>pred_prob):
                pred_prob = temp_prob
                pred_class = class_num
        print (class_prob)
        if pred_class==int(test_data[i][-1]):
            total_corr+=1
    return total_corr/(len(test_data)-1)
        
        
        
                        
                    
                
                
        



# In[13]:


def cal_feanum(file_name):
    res={}
    with open(file_name) as train_f:
        train_data=np.array(list(csv.reader(train_f)))
    for demen_num,demen_name in enumerate(train_data[0]):
        if demen_name!='Label' and demen_name!='No.'and demen_name!='Density'and demen_name!='SugerRatio':
            all_sample = train_data[1:,demen_num]
            #print(all_sample)
            res[demen_name]= np.unique(all_sample).shape[0];
    return res;
    


# In[15]:


fea_num = cal_feanum('train.csv')
#for fea in (fea_num).keys():
    #print (fea,'    ',fea_num[fea])


# In[63]:


def cal_prob_lap(file_name,train_data,laplase):
    with open(file_name) as test_f:
        test_data=list(csv.reader(test_f))
    total_corr= 0
    #print (test_data)
    for i in range(1,len(test_data)):
        class_prob={}
        pred_class=0
        pred_prob=0
        #Calculate the possibility for each class
        for class_num in train_data.keys():
            res=1
            
            #Calculate the likelihood on each demension
            for demen_num,demen_name in enumerate(test_data[0]):
                if demen_name!='Label' and demen_name!='No.':                    
                    x = float(test_data[i][demen_num])
                    #When the data follows continuous distribution
                    if demen_name=='Density' or demen_name=='SugerRatio' :
                        #Find the data in training dataset for the class on this demension
                        mean= train_data[class_num][demen_name]['mean']
                        std=train_data[class_num][demen_name]['std']
                        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
                        prob = (1 / (math.sqrt(2*math.pi) * std)) * exponent
                        
                        res=res*prob
                    #When the data follows discrete distribution
                    else:
                        x = int(test_data[i][demen_num])
                        sample_data = train_data[class_num][demen_name]['sample']
                        is_equal=(sample_data==x)
                        #if laplase smoothing is not applied 
                        if(laplase=='false'):
                            #print('this_num:%d all_num:%d' %(sum(is_equal.astype(int)),len(sample_data)))
                            prob = sum(is_equal.astype(int))/len(sample_data)
                        else:
                            #print('for demension:%s after_laplase:son:%d mother:%d' %(demen_name,sum(is_equal.astype(int))+1,(len(sample_data)+fea_num[demen_name])))
                            prob = (sum(is_equal.astype(int))+1)/(len(sample_data)+fea_num[demen_name])                       
                        res = res * prob   
            if laplase=='false':
                temp_prob=res*(len(train_dataset[class_num])/total_trainnum)
            else:
                temp_prob = res*((len(train_dataset[class_num])+1)/(total_trainnum+len(train_data.keys())))
            class_prob[class_num]=temp_prob
            if(temp_prob>pred_prob):
                pred_prob = temp_prob
                pred_class = class_num
        print ('The prediction result of the %dth data is %d'%(i,pred_class))
        if pred_class==int(test_data[i][-1]):
            total_corr+=1
    return total_corr/(len(test_data)-1)
        


# In[64]:
print ('With Laplase Smoothing: ')

rate = cal_prob_lap('test.csv',train_dataset,'true')
print ('The accuracy is %f '%(rate))

# In[65]:

print ('Without Laplase Smoothing: ')
rate2 = cal_prob_lap('test.csv',train_dataset,'false')
print ('The accuracy is %f '%(rate2))

