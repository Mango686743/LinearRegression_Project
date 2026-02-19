import numpy as np

class LinearRegression:
    
    def __init__(self,learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weight = 0
        self.bias = 0
    
    def fit(self,X,y):
        m_rows, n_cols = X.shape
        
        self.weight = np.zeros(n_cols)     
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X,self.weight) + self.bias
            
            dw = (1/m_rows)*np.dot(X.T,(y_pred-y))
            db = (1/m_rows)*np.sum(y_pred-y)
            
            self.weight = self.weight - self.lr*dw
            self.bias = self.bias - self.lr*db
            
    def predict(self,X):
        return np.dot(X,self.weight)+self.bias