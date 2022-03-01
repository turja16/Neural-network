import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return np.multiply(sigmoid(z),1-sigmoid(z))

def calc_gradient(W, b, X, delta, eta):
    W_grad = [None] * len(W)
    b_grad = [None] * len(b)
    W_grad = X.T.dot(delta)
    b_grad = delta.mean()
    return (W_grad, b_grad)

def weighted_sum(W, x, b):
    return  np.dot(W, x.T) + b

def cost(y_pred, y):
    result = np.multiply((y_pred - y),(y_pred - y))
    return result.mean()

def cost_grad(y_pred, y):
    return (y_pred -np.asmatrix(y))

def forward_pro(x, W, b):
    z = weighted_sum(W, x, b)
    y_pred = sigmoid(z)
    return y_pred

def back_prop(y_pred, W, y):
    cost = cost_grad(y_pred, y)
    delta = np.multiply(cost, 1-y_pred)
    return delta

def backPropagation_iter(X, y, W, b, eta):
    y_pred = forward_pro(X, W, b)
    delta = back_prop(y_pred.T, W, y.T)
    W_grad, b_grad = calc_gradient(W, b, X, delta, eta)
    W = W - eta*np.squeeze(W_grad.T, axis= 0)
    b = b - eta*b_grad
    MSE = cost(y_pred,y)
    return (W, b, y_pred, MSE)

if __name__ == '__main__':
    X = np.matrix('0.1 0.1; 0.1 0.9; 0.9 0.1; 0.9 0.9')
    y = np.matrix([0.1, 0.9, 0.9, 0.9])
    W = np.array([1.5, 3.8])
    b = np.array([1, 1, 1, 1])
    eta, MSE, itr = 0.1, 100, 0
    error_list, b_list, w0_list, w1_list = [], [], [], []
    while MSE >= 0.01:
        W, b, y_pred, MSE = backPropagation_iter(X, y, W, b, eta)
        error_list.append(MSE)
        b_list.append(b)
        w0_list.append(W.item(0, 0))
        w1_list.append(W.item(0, 1))
        itr = itr + 1
        print("MSE {0}",MSE)

plt.plot(list(range(itr)), error_list,'--r', label ='Error')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")

plt.title("Error Vs Iteration")
plt.show()
plt.close()
plt.xlabel("Iteration")
plt.ylabel("Bias/Weight")

plt.plot(list(range(itr)), w0_list, '-y', label = 'Weight 0')
plt.plot(list(range(itr)), w1_list, '-g', label = 'Weight 1')
plt.plot(list(range(itr)), b_list, '-r', label = 'Weight 3')
plt.legend(['Weight 0','Weight 1','Bias'])
plt.title("Weight/Bias Vs Iteration")
plt.show()
itr = itr + 1