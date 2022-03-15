import numpy
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return np.multiply(sigmoid(z),1-sigmoid(z))

def calc_gradient(W, b, a, delta, eta):
    W_grad = [None] * len(W)
    b_grad = [None] * len(b)

    for l in range(1, len(W)):
        chk_a = a[l-1].T
        chk_del = delta[l]
        chk_wgrad = chk_del.dot(chk_a)
        W_grad[l] = chk_wgrad
        b_grad[l] = delta[l]

    return (W_grad, b_grad)

def weighted_sum(W, a, b):
    result = np.dot(W,a) + b
    return result

def cost(a, y):
    result = 0
    for i in range(n):
        result += ((a[i] - y[i]) ** 2)
    return result/n

def cost_grad(a, y):
    return (a - np.asmatrix(y))

def forward_pro(x, W, b):
    a = [None] * len(layer_sizes)
    z = [None] * len(layer_sizes)

    a[0] = x[0].T

    for l in range(1, len(a)):
        z[l] = weighted_sum(W[l], a[l-1], b[l])
        a[l] = sigmoid(z[l])

    return (a, z)

def back_prop(a, z, W, y):
    delta = [None] * len(layer_sizes)
    end_node = len(a) - 1
    chk_grad = cost_grad(a[end_node],y)
    chk_sig = sigmoid_prime(z[end_node])
    delta[end_node] = np.multiply(cost_grad(a[end_node],y), sigmoid_prime(z[end_node]))

    for l in reversed(range(1, end_node)):
        chk_w = W[l+1].T
        chk_delta = delta[l+1]
        chk_sig = sigmoid_prime(z[l])
        chk_dot = chk_w.dot(chk_delta)
        chk_del = np.multiply(chk_sig,chk_dot)
        delta[l] = chk_del

    return delta

def backPropagation_iter(X, y, W, b, eta):

    y_pred = [None] * len(y)

    for i in range(n):
        a, z = forward_pro(X[i,:], W, b)
        y_pred[i] = np.max(a[-1])
        delta = back_prop(a, z, W, y[i])
        W_grad, b_grad = calc_gradient(W, b, a, delta, eta)

        if i == 0:
            W_grad_sum = W_grad
            b_grad_sum = b_grad
        else:
            for l in range(1, len(W_grad)):
                W_grad_sum[l] += W_grad[l]
                b_grad_sum[l] += b_grad[l]


    for l in range(1, len(W)):
        W[l] = W[l] - (eta/n) * W_grad_sum[l]
        b[l] = b[l] - (eta/n) * b_grad_sum[l]

    MSE = cost(y_pred, y)

    return (W, b, y_pred, MSE)

if __name__ == '__main__':
    X = np.matrix('0.1 0.1; 0.1 0.9; 0.9 0.1; 0.9 0.9')
    y = np.array([0.1, 0.9, 0.9, 0.1])
    layer_sizes = [X.shape[1],2,1]
    W = [None] * len(layer_sizes)
    b = [None] * len(layer_sizes)

    for l in range(1, len(layer_sizes)):
        W[l] = np.random.random((layer_sizes[l],layer_sizes[l-1]))
        b[l] = np.random.random((layer_sizes[l],1))

    eta, MSE, itr, n = 0.1, 100, 0, X.shape[0]
    error_list, b_list, w_list = [], [], []
    b10_list, b11_list, b20_list = [],[],[]
    w00_list,w01_list,w10_list,w11_list = [], [], [], []
    w001_list, w011_list= [], []
    while MSE >= 0.01:
        W, b, y_pred, MSE = backPropagation_iter(X, y, W, b, eta)
        error_list.append(MSE)
        b_list.append(b)

        w00_list.append(W[1].item(0,0))

        w01_list.append(W[1].item(0,1))
        w10_list.append(W[1].item(1,0))
        w11_list.append(W[1].item(1,0))

        b10_list.append(b[1].item(0,0))
        b11_list.append(b[1].item(1,0))


        w001_list.append(W[2].item(0,0))

        w011_list.append(W[2].item(0,1))


        b20_list.append(b[2].item(0, 0))
        itr = itr + 1
        #print("MSE {0}",MSE)

plt.plot(list(range(itr)), error_list,'--r', label ='Error')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Error")

plt.title("Error Vs Iteration")
plt.show()
plt.close()



plt.plot(list(range(itr)), w00_list, '-y', label = 'Weight 00')
plt.title("Input Layer Weight X0 Vs Iteration")
plt.show()
plt.close()
plt.plot(list(range(itr)),  w01_list, '-g', label = 'Weight 01')
plt.title("Input Layer  Weight Y0  Vs Iteration")
plt.show()
plt.close()
plt.plot(list(range(itr)),  w10_list, '-b', label = 'Weight 10')
plt.title("Input Layer Weight X1  Vs Iteration Vs Iteration")
plt.show()
plt.close()
plt.plot(list(range(itr)),  w11_list, '-r', label = 'Weight 11')
plt.title("Input Layer Weight Y1  Vs Iteration")
plt.show()
plt.close()
plt.plot(list(range(itr)),  b10_list, '-r', label = 'Weight 11')
plt.title("Hidden Layer 1 Bias  Vs Iteration")
plt.show()
plt.close()
plt.plot(list(range(itr)),  b11_list, '-r', label = 'Weight 11')
plt.title("Hidden Layer 2 Bias  Vs Iteration")
plt.show()
plt.close()

plt.plot(list(range(itr)), w001_list, '-y', label = 'Weight 00')
plt.title("Hidden Layer Weight 00 Vs Iteration")
plt.show()
plt.close()
plt.plot(list(range(itr)),  w011_list, '-g', label = 'Weight 01')
plt.title("Hidden Layer  Weight 01  Vs Iteration")
plt.show()
plt.close()

plt.plot(list(range(itr)),  b20_list, '-r', label = 'Weight 11')
plt.title("output Layer  Bias  Vs Iteration")
plt.show()
plt.close()