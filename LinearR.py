import numpy as np
from numpy.linalg import inv
from sklearn.metrics import accuracy_score
import pandas as pd


colnames=['f1', 'f2', 'f3', 'f4','label'] 
# this is my local path for Iris data D:\Spring 2023\Machine Learning\iris.data' 
df = pd.read_csv('D:\MSCS_UTA_RA\Spring 2023\Machine Learning\Assignments and Projects\P1\iris.data',names=colnames,header=None,index_col=False)

# setting the lables into numericals
df.replace("Iris-setosa",1,inplace=True)
df.replace("Iris-versicolor",2,inplace=True)
df.replace("Iris-virginica",3,inplace=True)

# X will be our features and y will be lables
X = df.iloc[:,:4]
y = df.iloc[:, 4]


dim = X.shape
dim[0]

#Input fold value to perform k-fold cross validation
fold_num = int(input("enter the kfold number\n"))

#Below is the Linear regression function which will calculation the y=b0 +bX and 
#returns the accuracy which is array contaning each fold accuracy value
def linear_regression(X,y,X1,y1):
  #print("Hii im in linear regression")
  
  b = inv(X.T.dot(X)).dot(X.T).dot(y)
  
  #print("beta = ",b)
  x_mean = np.mean(X, axis=0)
  x_mean
  y_mean = np.mean(y)
  

  b0 = y_mean - b[0]*x_mean[0] - b[1]*x_mean[1] - b[2]*x_mean[2] - b[3]*x_mean[3]
  #print("bo is ",b0)
  
  test_data = b0 + X1.dot(b)
  # now predicting the test data
  test_result = []
  #print(test_data)
  for i in test_data.index:
  
    test_result.append(predict(test_data[i]))
    #print(test_data)
  
  accuracy = accuracy_score(y1, test_result)
  return accuracy

#the predict function below will choose the result label values based on the predict range of values 
def predict(y):
  if y>0 and y<1.5:
    label = 1
  elif y>1.5 and y<2.3:
    label=2
  elif y>2.3 and y<4:
    label =3
  
  return label

#below is the K-fold function which takes 3 parameters 
# Parameter 1 : Number of folds 
# Parameter 2 : is our features 
# Parameter 3 label values
def kfold(num,X,y):
  
  train_foldX = pd.DataFrame()
  test_foldX = pd.DataFrame()
  train_foldy = pd.DataFrame()
  test_foldy = pd.DataFrame()
  accuracy = []*num
  foldval = dim[0]//num
  
  ini = 0
  for i in range(0,num):
    start = ini
    end = start+foldval
    if i==num-1:
      end = dim[0]-1
      train_foldX =X[:start]
      train_foldy = y[:start]
    elif i==0:
      train_foldX =X[end:]
      train_foldy = y[end:]
    else:
      end = start +foldval
      train_foldX = np.concatenate([X[:start], X[end:]])
      train_foldy = np.concatenate([y[:start], y[end:]])
    #actual code to do
    test_foldX = X[start:end]
    test_foldy = y[start:end]
    
    #print(start,end)
    accuracy.append(linear_regression(train_foldX,train_foldy,test_foldX,test_foldy))
    #print(accuracy)
    #print(train_foldX.shape,train_foldy.shape,test_foldX.shape,test_foldy.shape)
    

    ini +=foldval
  return accuracy

finalres = kfold(fold_num,X,y)
print("The accuracy of each fold is ",finalres)
print("The avg accuracy of the model is",sum(finalres)/len(finalres))
