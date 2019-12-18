import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('ex5data1.mat')
for k in data:
    print(k)

X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

plt.figure(figsize=(8, 6))
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Figure 1: Data')
plt.plot(X, y, 'rx')

# Insert the 1's column.
X_with_1s = np.insert(X, 0, 1, axis=1)
Xval_with_1s = np.insert(Xval, 0, 1, axis=1)

# Create a function to compute cost and gradient.
def linearRegCostFunction(X, y, theta, lambda_coef):
    """
    Computes cost and gradient for regularized
    linear regression with multiple variables.
    Args:
        X: array (m, number of features+1)
        y: array (m, 1)
        theta: array (number of features+1, 1)
        lambda_coef: float
    Returns:
        J: float
        grad: vector array
    """
    # Get the number of training examples, m.
    m = len(X)
    
    # Ensure theta shape(number of features+1, 1).
    theta = theta.reshape(-1, y.shape[1])
    
    #############################################################
    ###################### Cost Computation #####################
    #############################################################
    # Compute the cost.
    unreg_term = (1 / (2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    
    # Note that we should not regularize the theta_0 term!
    reg_term = (lambda_coef / (2 * m)) * np.sum(np.square(theta[1:len(theta)]))
    
    cost = unreg_term + reg_term
    
    #############################################################
    #################### Gradient Computation ###################
    #############################################################
    # Initialize grad.
    grad = np.zeros(theta.shape)

    # Compute gradient for j >= 1.
    grad = (1 / m) * np.dot(X.T, np.dot(X, theta) - y) + (lambda_coef / m ) * theta
    
    # Compute gradient for j = 0,
    # and replace the gradient of theta_0 in grad.
    unreg_grad = (1 / m) * np.dot(X.T, np.dot(X, theta) - y)
    grad[0] = unreg_grad[0]

    return (cost, grad.flatten())

# Initialize theta at [1, 1].
theta = np.array([[1], [1]])

# Run the cost function with theta [1, 1] and lambda_coef = 1.
(cost, grad) = linearRegCostFunction(X_with_1s, y, theta, 1)
print('Cost and Gradient at theta = [1, 1]:', (cost, grad))

from scipy.optimize import minimize

def trainLinearReg(X, y, lambda_coef):
    """
    Trains linear regression using the dataset (X, y)
    and regularization parameter lambda_coef.
    Returns the trained parameters theta.
    Args:
        X: array (m, number of features+1)
        y: array (m, 1)
        lambda_coef: float
    Returns:
        theta: array (number of features+1, )
    """
    # Initialize Theta.
    initial_theta = np.zeros((X.shape[1], 1))
    
    # Create "short hand" for the cost function to be minimized.
    def costFunction(theta):
        return linearRegCostFunction(X, y, theta, lambda_coef)
    
    print("--------------------------------------------")
    print(len(X))
    print(len(X[0]))
    # Now, costFunction is a function that takes in only one argument.
    results = minimize(fun=costFunction,
                       x0=initial_theta,
                       method='CG',
                       jac=True,
                       options={'maxiter':200})
    theta = results.x

    return theta

# Train linear regression with lambda_coef = 0.
theta = trainLinearReg(X_with_1s, y, 0)
theta

plt.figure(figsize=(8, 6))
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Figure 2: Linear Fit')
plt.plot(X, y, 'rx')
plt.plot(X, np.dot(np.insert(X, 0, 1, axis=1), theta), '--')

# Create a function that generates the errors.
def learningCurve(X, y, Xval, yval, lambda_coef):
    """
    Generates the train and cross validation set errors needed 
    to plot a learning curve. In particular, it returns two
    vectors of the same length - error_train and error_val.
    Then, error_train(i) contains the training error for
    i examples (and similarly for error_val(i)).
    Note:
        We should evaluate the training error on the first
        i training examples (i.e., X(1:i, :) and y(1:i)), but
        for the cross-validation error, we should instead
        evaluate on the entire cross validation set (Xval, yval).
    Args:
        X: array (m, number of features+1)
        y: array (m, 1)
        Xval: array (s, number of features+1)
        yval: array (s, 1)
        lambda_coef: float
    Returns:
        error_train: array (m, 1)
        error_val: array (m, 1)
    """
    # Get the number of training examples, m.
    m = len(X)
    
    # Initialize train and cross validation set errors.
    error_train = np.zeros((m, 1))
    error_val   = np.zeros((m, 1))
    
    for i in range(1, m+1):
        # Train with different subsets of the training
        # set X to obtain the theta parameters.
        theta = trainLinearReg(X[:i], y[:i], lambda_coef)
        
        # Compute train/cross validation errors, storing
        # the result in error_train and error_val.
        # Note that to compute the errors, we should call
        # the function with the lambda_coef set to 0.
        error_train[i-1] = linearRegCostFunction(X[:i], y[:i], theta, 0)[0]
        error_val[i-1] = linearRegCostFunction(Xval, yval, theta, 0)[0]
        
    return error_train, error_val

m = len(X)

# Get the errors with lambda_coef set to 0.
error_train, error_val = learningCurve(X_with_1s, y,
                                       Xval_with_1s, yval, 0)

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('{}\t\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))

plt.figure(figsize=(8, 6))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.title('Figure 3: Linear regression learning curve')
plt.plot(range(1,m+1), error_train, 'b', label='Train')
plt.plot(range(1,m+1), error_val, 'g', label='Cross Validation')
plt.legend()   
plt.show()

# Create a function that maps the original X into its higher powers.
def polyFeatures(X, p):
    """
    Takes a data set X and maps each example into its polynomial
    features, where X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p]
    Args:
        X: array (m, 1)
        p: int > 1
    Returns:
        X_poly: array (m, p)
    """
    # Initialize X_poly.
    X_poly = X

    # Iterate over the polynomial power.
    for i in range(1, p):
        # Add the i-th power column in X.
        X_poly = np.column_stack((X_poly, np.power(X, i+1)))   

    return X_poly

# Create a function that normalizes the features of X.
def featureNormalize(X):
    """
    Returns a normalized version of X where the mean value
    of each feature is 0 and the standard deviation is 1.
    Args:
        X: array (m, number of features)
    Returns:
        X: array (m, number of features)
    """
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    
    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm / sigma
    
    return X_norm, mu, sigma

p = 8

# Map X onto Polynomial Features and Normalize.
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly) # Normalize
X_poly = np.insert(X_poly, 0, 1, axis=1) # Add Ones

# Map X_poly_test and normalize (using mu and sigma).
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.insert(X_poly_test, 0, 1, axis=1) # Add Ones

# # Map X_poly_val and normalize (using mu and sigma).
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.insert(X_poly_val, 0, 1, axis=1) # Add Ones

print('Normalized Training Example 1:')
print(X_poly[1, :])

# Create a function to plot a polynomial regression fit.
def plotFit(min_x, max_x, mu, sigma, theta, p):
    """
    Plots the learned polynomial fit with power p
    and feature normalization (mu, sigma).
    Also works with linear regression.
    Args:
        X: array (m, number of features)
    Returns:
        X: array (m, number of features)
    """
    # We plot a range slightly bigger than the min and max
    # values to get an idea of how the fit will vary outside
    # the range of the data points.
    x = np.array(np.arange(min_x - 15, max_x + 25, 0.05))

    # Map the X values.
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma

    # Add ones.
    X_poly = np.insert(X_poly, 0, 1, axis=1)

    # Plot
    plt.plot(x, np.dot(X_poly, theta), '--')

# Train linear regression with lambda_coef = 0.
theta = trainLinearReg(X_poly, y, 0)

plt.figure(figsize=(8, 6))
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Figure 4: Polynomial fit, $\lambda$ = 0')
plt.plot(X, y, 'rx')
plotFit(min(X), max(X), mu, sigma, theta, p)
plt.show()
# Get the errors with lambda_coef set to 0.
#error_train, error_val = learningCurve(X_poly, y,
#                                       X_poly_val, yval, 0)

#plt.figure(figsize=(8, 6))
#plt.xlabel('Number of training examples')
#plt.ylabel('Error')
#plt.title('Figure 5: Polynomial learning curve, $\lambda$ = 0')
#plt.plot(range(1,m+1), error_train, 'b', label='Train')
#plt.plot(range(1,m+1), error_val, 'g', label='Cross Validation')
#plt.legend()
#plt.show()    