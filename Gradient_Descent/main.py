import numpy as np

def batch_gradient_descent(X, y, learning_rate, num_iterations):
    m = X.shape[0]
    n = X.shape[1]
    
    theta = np.zeros(n)
    
    for i in range(num_iterations):
        predictions = np.dot(X, theta)
        gradient = (1/m) * np.dot(X.T, (predictions - y))
        theta = theta - learning_rate * gradient
        
        if i % 100 == 0:
            cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
            print(f"Iteration {i}: Cost = {cost}")
    
    return theta

def mini_batch_gradient_descent(X, y, learning_rate, num_iterations, batch_size):
    m = X.shape[0]
    n = X.shape[1]
    
    theta = np.zeros(n)
    
    for i in range(num_iterations):
        random_indices = np.random.choice(m, batch_size, replace=False)
        X_batch = X[random_indices]
        y_batch = y[random_indices]
        
        predictions = np.dot(X_batch, theta)
        gradient = (1/batch_size) * np.dot(X_batch.T, (predictions - y_batch))
        
        theta = theta - learning_rate * gradient
        
        if i % 100 == 0:
            full_predictions = np.dot(X, theta)
            cost = (1/(2*m)) * np.sum((full_predictions - y) ** 2)
            print(f"Iteration {i}: Cost = {cost}")
    
    return theta

def stochastic_gradient_descent(X, y, learning_rate, num_iterations):
    m = X.shape[0]
    n = X.shape[1]
    
    theta = np.zeros(n)
    
    for i in range(num_iterations):
        random_index = np.random.randint(m)
        X_i = X[random_index, :].reshape(1, n)
        y_i = y[random_index].reshape(1)
        
        prediction = np.dot(X_i, theta)
        gradient = np.dot(X_i.T, (prediction - y_i))
        
        theta = theta - learning_rate * gradient
        
        if i % 100 == 0:
            full_predictions = np.dot(X, theta)
            cost = (1/(2*m)) * np.sum((full_predictions - y) ** 2)
            print(f"Iteration {i}: Cost = {cost}")
    
    return theta

##### Testing #####
X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]]) 
y = np.array([5, 7, 9, 11])

learning_rate = 0.01
num_iterations = 1000
batch_size = 2

print("Batch Gradient Descent")
best_theta_batch = batch_gradient_descent(X, y, learning_rate, num_iterations)

# learning_rate = 0.50
# best_theta = batch_gradient_descent(X, y, learning_rate, num_iterations)
# print("Final parameters:", best_theta)

# The learning rate of 0.50 was too large, causing the gradient descent algorithm to overshoot 
# the minimum of the cost function. When the weights grow too large, squaring these large values 
# in the cost function leads to an overflow, and the resulting computations can't be handled by 
# floating-point precision, leading to inf (infinity) and NaN. Tried a scaling technique,
# but absolutely failed. 

print("\nMini-batch Gradient Descent")
best_theta_mini_batch = mini_batch_gradient_descent(X, y, learning_rate, num_iterations, batch_size)

print("\nStochastic Gradient Descent")
best_theta_sgd = stochastic_gradient_descent(X, y, learning_rate, num_iterations)