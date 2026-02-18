import numpy as np

class LinearRegression:         #Predict y = wx+b
    def __init__(self, learning_rate = 0.01, n_iters = 1000 ):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weight = None      #Trọng số: a
        self.bias = None        #Độ lệch: b
        
    def fit(self,X,y):                  #Ma trận X và y
        #Training find w,b
        m_rows, n_columns = X.shape     #Ma trận X có m hàng n cột        
        
        #Khởi tạo 0
        self.weight = np.zeros(n_columns)           #Tạo 1 cột vecto 0    
        self.bias = 0
        
        #Học 1000 lần
        for _ in range(self.n_iters):
            y_pred = np.dot(X,self.weight) + self.bias
            
            #Gradient descent(Đạo hàm) = Hàm Loss
            dw = (1/m_rows) * np.dot(X.T,y_pred-y)
            db = (1/m_rows) * np.sum(y_pred-y)       
            
            #Update w và b -> fit
            self.weight = self.weight - self.lr*dw
            self.bias = self.bias - self.lr*db
    
    def predict(self, X):
        #predict
        return np.dot(X, self.weight) + self.bias