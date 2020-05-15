import numpy as np

class LinearRegression:
    def __init__(self):
        # attributes
        self.weights = None
        self.bias = None

        self.losses = []

    def __slope(self, x, w, b):
        '''
		performs slope calculation y = m * x + b
		x = Input Features
		w = Weights
		b = Bias
		'''
        return np.matmul(x, w) + b

    def __loss(self, x, y, w, b):
        '''
		computes Mean Sqared Error for given X & y per iteration
		'''
        loss = 0
        pred = self.__slope(x, w, b)
        loss = np.square((y - pred)).mean(axis=0)
        return loss

    def __optimizer(self, x, y, w, b, learning_rate):
        '''
		performs Gradient Descent to optimize Weights & Bias paramters per iteration
		'''
        dw, db = 0, 0

        dw += - (2 * np.dot((y - self.__slope(x, w, b)).transpose(), x)).reshape(w.shape)
        db += - (2 * (y - self.__slope(x, w, b))).mean().reshape(b.shape)

        w -= learning_rate * dw
        b -= learning_rate * db

        return w, b    

    def fit(self, X, y, epochs=30, learning_rate=0.1, batch_size=8, verbose=1):
        '''
		Training function
		'''
		# Initialize/Sample weights and biases from a random normal distribution
		# Xavier Initialization
		# Square Root(6 / (1.0 + input features + output features))
        lim = np.sqrt(6.0 / (X.shape[0] + X.shape[1] + 1.0))

        w = np.random.uniform(low = -lim, high = lim, size=(X.shape[1], 1))
        b = np.random.uniform(low = -lim, high = lim, size=1)

        num_samples = len(X)
        
        # Train the model for given epochs in batches
        for i in range(epochs):
            # create batches
            for offset in range(0, num_samples, batch_size):
                # create batches 
                end = offset + batch_size
                batch_x, batch_y = X[offset:end], y[offset:end]

                # calculate loss
                loss = self.__loss(batch_x, batch_y, w, b)

                # perform Gradient Descent to optimize Weights & Biases	
                w, b = self.__optimizer(batch_x, batch_y, w, b, learning_rate)
            
            # store losses as an array
            self.losses.append(loss)

            # Display training loss based on interval value
            if((i==0) or (i==(epochs-1) or (i % verbose) == 0)):
                print(f"Epoch {i+1}, Loss: {loss[0]:.4f}")

        self.weights = w
        self.bias = b        

    def predict(self, x):
        '''
		returns predicted values when input with new data points
		'''
        if self.weights is not None:
            return np.matmul(x, self.weights) + self.bias

        else:
            lim = np.sqrt(6.0 / (X.shape[0] + X.shape[1] + 1.0))

            w = np.random.uniform(low = -lim, high = lim, size=(X.shape[1], 1))
            b = np.random.uniform(low = -lim, high = lim, size=1)[0]

            return np.matmul(x, w) + b

class LogisticRegression:
    def __init__(self):
        # attributes
        self.weights = None
        self.bias = None

        self.losses = []

    def __slope(self, x, w, b):
        '''
		performs slope calculation y = m * x + b
		x = Input Features
		w = Weights
		b = Bias
		'''
        return np.matmul(x, w) + b

    def __sigmoid(self, x):
        '''
		performs Sigmoid/Logistic fucntion over input value
		'''
        return 1 / (1 + np.exp(-x))     

    def __loss(self, x, y, w, b):
        '''
		returns binary cross-entropy value
		'''
        z = self.__sigmoid(self.__slope(x, w, b))
        loss = (y * np.log(1e-15 + z)) + ((1 - y) * np.log(1-(z + 1e-15)))
        return - loss.mean(axis=0)    

    def __optimize(self, x, y, w, b, learning_rate):
        '''
		performs Gradient Descent to optimize Weights & Bias paramters per iteration
		'''
        dw, db = 0, 0
        
        z = self.__sigmoid(self.__slope(x, w, b))
        grad_loss = (z - y)

        dw = np.dot(x.transpose(), grad_loss)
        db = np.mean(grad_loss * x.shape[0]) 

        w -= learning_rate * dw.reshape(w.shape)
        b -= learning_rate * db.reshape(b.shape)

        return w, b    

    def fit(self, X, y, epochs=30, batch_size=8, learning_rate=0.1, verbose=1):
        '''
		Training function
		'''
		# Initialize/Sample weights and biases from a random normal distribution
		# Xavier Initialization
		# Square Root(6 / (1.0 + input features + output features))
        lim = np.sqrt(6.0 / (X.shape[0] + X.shape[1] + 1.0))

        w = np.random.uniform(low = -lim, high = lim, size=(X.shape[1], 1))
        b = np.random.uniform(low = -lim, high = lim, size=1)

        num_samples = len(X)
        
        # Train the model for given epochs in batches
        for i in range(epochs):
            # create batches
            for offset in range(0, num_samples, batch_size):
                # create batches 
                end = offset + batch_size
                batch_x, batch_y = X[offset:end], y[offset:end]

                # calculate loss
                loss = self.__loss(batch_x, batch_y, w, b)

                # perform Gradient Descent to optimize Weights & Biases	
                w, b = self.__optimize(batch_x, batch_y, w, b, learning_rate)
            
            # store losses as an array
            self.losses.append(loss)

            # Display training loss based on interval / verbose value
            if((i==0) or (i==(epochs-1) or (i % verbose) == 0)):
                print(f"Epoch {i+1}, Loss: {loss[0]:.4f}")

        self.weights = w
        self.bias = b        

    def predict(self, X, threshold=0.5):
        '''
		returns predicted class values when input with new data points
		'''
        if self.weights is not None:
            return self.__sigmoid(np.matmul(X, self.weights) + self.bias) >= threshold

        else:
            lim = np.sqrt(6.0 / (X.shape[0] + X.shape[1] + 1.0))

            w = np.random.uniform(low = -lim, high = lim, size=(X.shape[1], 1))
            b = np.random.uniform(low = -lim, high = lim, size=1)[0]

            return self.__sigmoid(np.matmul(X, w) + b) >= threshold