
import numpy as np
import pandas as pd
import copy

def IsScalar(x):
    if type(x) in (list, np.ndarray,):
        return False
    else:
        return True

def Thresh(x):
    if IsScalar(x):
        val = 1 if x>0 else -1
    else:
        val = np.ones_like(x)
        val[x<0] = -1.
    return val

def Energy(W, X):
    Y = np.asmatrix(X)
    test = Y @ W
    E = - 0.5 * (Y @ W)@Y.T
    return E


def Update(W, x):
    xnew = x @ W
    return Thresh(xnew)

def Iterate_Network( W, x_orig, isSync):
    if isSync:
        x = copy.deepcopy(x_orig)
        e = [1000,]
        e_new = Energy(W,x)
        n_iters = 0
        while e_new < e:
            x_new = Update(W, x)
            #print(Hamming(x, x_new))
            e = e_new
            e_new = Energy(W,x_new)
            x = x_new
            n_iters = n_iters +1


        print(x_new)
        print(n_iters)
        return x_new, n_iters, e_new

    else:
        x = copy.deepcopy(np.asmatrix(x_orig))
        n_iters = 0
        e = [10000,]
        e_new = Energy(W,x)
        while e_new < e:
            node_idx = list(range(np.shape(x)[1]))
            np.random.shuffle(node_idx)
            for k in node_idx:
                ic = x @ W[:, k]
                x[0, k] = Thresh(ic)
            e = e_new
            e_new = Energy(W, x)
            n_iters = n_iters + 1

        return x, n_iters, e_new

def Hamming(x, y):
    d = []
    for xx, yy in zip(x, y):
        dd = 0.
        for xxx, yyy in zip(xx, yy):
            if xxx == 1 and yyy != 1:
                dd += 1.
            elif yyy == 1 and xxx != 1:
                dd += 1.
        d.append(dd)
    return d




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def gen_bits_by_bitcount(value):
    return np.array([list(np.binary_repr(x, value)) for x in range(2**value)], dtype=int)


def cal_weights(X):
    N_0 = np.shape(X)[0]
    N_1 = np.shape(X)[1]
    W = (X.T @ X)  - N_0*np.eye(N_1)
    return W



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    X = np.array([   [1, 1, 1, -1, 1, -1, -1, 1, -1],
        [1, -1, 1, 1, -1, 1, 1, 1, 1],
                      ])


    W = cal_weights(X)
    print(W)
    test_data = gen_bits_by_bitcount(9)
    output = []
    for count in range(512):
        x_async, n_iters_async, e_async = Iterate_Network(W, test_data[count], False)
        x_sync, n_iters_sync, e_sync = Iterate_Network(W, test_data[count], True)
        output.append((test_data[count],x_sync, n_iters_sync, e_sync,x_async, n_iters_async, e_async))

    df = pd.DataFrame(output, columns=['Inital State', 'Final State', 'Iteration(Sync)', 'Energy(Sync)', 'Final State', 'Iteration(ASync)', 'Energy(ASync)'])

    #print(df)
    df.to_csv('sample.csv')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
