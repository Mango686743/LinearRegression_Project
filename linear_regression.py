import numpy as np

class LinearRegression:         #y = w(n).x^(n) + w(n-1).x^(n-1) + ... + w.x + b
    def __init__(self, learning_rate = 0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weight = 0         #matrix of w
        self.bias = 0           #b
        
    def fit(self,X,y):          #Matrix of weight X; Matrix of result y
        #Matrix size
        m_rows, n_cols = X.shape
        
        #Initialization
        self.weight = np.zeros(n_cols)      #Vector 0 size nx1
        self.bias = 0                       #Auto transform to vecto -> Calculating
        
        #Calculating
        for _ in range(self.n_iters):
            #Proj of y -> Find Least Square
            y_proj = np.dot(X,self.weight) + self.bias      
            
            #Gradient descent by Loss Fuction
            dw = (1/m_rows)*np.dot(X.T,(y_proj-y))
            db = (1/m_rows)*np.sum(y_proj-y)
        
            #Machine learning for finding fitter W and B
            self.weight -= self.lr*dw
            self.bias   -= self.lr*db
            
    def predict(self,X):
        return np.dot(X,self.weight) + self.bias        