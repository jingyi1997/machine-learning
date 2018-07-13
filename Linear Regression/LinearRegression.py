
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split  
import numpy as np
import math


# In[20]:


boston = load_boston()

data = np.array(boston['data'])    
num_train = data.shape[0]
target = np.array(boston['target'])
mean = np.mean(data,axis = 0)
std = np.std(data, axis = 0)
data = (data - mean)/std
data = np.column_stack((data,np.ones(num_train)))


# In[21]:


print()


# In[45]:


X_train2,X_test,y_train2, y_test = train_test_split(data, target,  test_size = 0.2, random_state = 0)  
X_train,X_val,y_train, y_val = train_test_split(X_train2, y_train2,  test_size = 0.1, random_state = 0)  
print('Training set shape:',X_train.shape)
print('Test set shape:',X_test.shape)
print('Validation set shape:',X_val.shape)
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)


# In[46]:



class LinearRegression(object):
   def __init__(self):
       self.W = None
   def loss(self, X, y, reg):
       """
       Compute the loss function and its derivative. 

       Inputs:
       - X: A numpy array of shape (N, D) containing  N
         data points; each point has dimension D.
       - y: A numpy array of shape (N,) containing targets for the set.
       - reg: (float) regularization strength.

       Returns: A tuple containing:
       - loss as a single float
       - gradient with respect to self.W; an array of the same shape as W
       """
       num_train = X.shape[0]
       
       pred = X.dot(self.W)
       
       #the mean square loss
       loss_ms = ((pred - y).T).dot((pred - y))/num_train/2
       #the regularization loss
       reg_W = self.W[:-1]
       loss_reg = reg*np.sum(reg_W*reg_W)
       #the total loss 
       loss = loss_ms  + loss_reg
       grad_ms = (X.T).dot(pred-y)/num_train   
       grad_reg = 2*reg*self.W
       grad_reg[-1] = 0
       grad = grad_reg + grad_ms
       return loss,grad
   def train (self, X_train, y_train,X_val,y_val, learning_rate, reg, num_iters):
       """
       Train this linear classifier using batch gradient descent.

       Inputs:
       - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
       - y: A numpy array of shape (N,) containing target; 
       - learning_rate: (float) learning rate for optimization.
       - reg: (float) regularization strength.
       - num_iters: (integer) number of steps to take when optimizing
     
       """
       num_train, dim = X_train.shape
       num_val = X_val.shape[0]
       if self.W is None:
         # lazily initialize W
         self.W = 0.001 * np.random.randn(dim)

       # Run batch gradient descent to optimize W
       loss_history = []
       train_rmse_history = []
       val_rmse_history = []
       for it in range(num_iters):
         # evaluate loss and gradient
         loss, grad = self.loss(X_train, y_train, reg)
         
         loss_history.append(loss)
         # perform parameter update
         # Update the weights using the gradient and the learning rate.          #
         self.W = self.W - grad*learning_rate
         if it % 1000== 0:
               # Check accuracy
               y_diff_train = self.predict(X_train) - y_train
               train_rmse = math.sqrt(np.sum(y_diff_train.T.dot(y_diff_train))/num_train)
               
               y_diff_val = self.predict(X_val) - y_val
               val_rmse = math.sqrt(np.sum(y_diff_val.T.dot(y_diff_val))/num_val)
        
               train_rmse_history.append(train_rmse)
               val_rmse_history.append(val_rmse)
               #print('iteration %d / %d: loss %f training_rmse: %f val_rmse: %f' % (it, num_iters, loss,train_rmse,val_rmse))
       return train_rmse,val_rmse       
    
   def predict(self, X):
       
       """
       Use the trained weights of this linear classifier to predict values for
       data points.

       Inputs:
       - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.

       Returns:
       - y_pred: Predicted values for the data in X. y_pred is a 1-dimensional
         array of length N.
       """
       y_pred = np.zeros(X.shape[0]) 
       y_pred = X.dot(self.W)
       return y_pred

 
       
       
   


# In[47]:


classifier = LinearRegression()


# In[59]:


for learning_rate in [1e-5,1e-6,1e-7]:
    for reg in [0.1,0.15,0.2,0.25,0.3]:        
        train,val =  classifier.train(X_train, y_train, X_val, y_val,
                    learning_rate,
                    reg,num_iters=100000)
        print('learning rate: %f, reg: %f, train_rmse:%f ,val_rmse:%f '%(learning_rate, reg, train,val))


# In[62]:


print('The value of the parameter is',classifier.W)


# In[63]:


num_test = X_test.shape[0]
y_diff_test = classifier.predict(X_test) - y_test
test_rmse = math.sqrt(np.sum(y_diff_test.T.dot(y_diff_test))/num_test)


# In[64]:


print('test_rmse: %f' %(test_rmse))


	

