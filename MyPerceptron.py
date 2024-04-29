import numpy as np
import pandas as pd

class MyPerceptron:

    def __init__(self,learning_rate=0.01,n_iters=10): #constructor
        self.lr=learning_rate
        self.no_of_iters=n_iters
        self.weights=None
        self.bias=None

    def fit(self,X,y):

        self.weights = np.zeros(X.shape[1])
        self.bias=0
        for j in range(self.no_of_iters):
            for i in range(X.shape[0]): 

                y_pred = self.actv_func(np.dot(self.weights,X[i]) + self.bias)

                self.weights=self.weights + self.lr * (y[i] - y_pred)*X[i]
                self.bias = self.bias + self.lr*(y[i] - y_pred)
        
        print("Training Complete")

    def actv_func(self,actv):
        if actv>=0:
            return 1
        else:
            return -1

    def predict(self,X):
        
        y_pred=[]

        for i in range(X.shape[0]):
            k=self.actv_func(np.dot(self.weights,X[i]) + self.bias)
            y_pred.append(k)

        return np.array(y_pred)
    
#reading and splitting the file
data = pd.read_excel('PLA_Data1.xlsx')

X=data.iloc[:,0:-1].values
y=data.iloc[:, -1].values

split = int(0.75*len(X))
X_train=X[:split]
X_test=X[split:]

y_train=y[:split]
y_test=y[split:]

#creating the perceptron
clf =  MyPerceptron()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(y_pred)
print(y_test)
accuracy_score = np.mean(y_pred == y_test)
print("Accuracy: ",accuracy_score)
