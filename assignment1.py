import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import math

# Seed a random number generator
seed = 12345
# rng = np.random.RandomState(seed)

'''
EXERCISE 1
'''
def process_data():
    amp_data = scipy.io.loadmat('amp_data.mat')
    amp_data_keys = amp_data['amp_data']
    # print(amp_data_keys)

    # x_grid = np.arange(start=0, stop=len(amp_data_keys), step=1)
    # plt.plot(x_grid, amp_data_keys,'b-')          # linear plot
    # plt.hist(amp_data_keys, normed=True, bins=16)   # histogram
    # plt.show()

    columns = 21
    rows = len(amp_data_keys) // columns
    data_set = amp_data_keys[:(rows*columns)]
    data_set = data_set.reshape((rows, columns))
    shuffled_data_set = shuffle_matrix(data_set)

    print (len(amp_data_keys))
    print (len(data_set))

    X_shuffle_train, X_shuffle_val, X_shuffle_test, y_shuffle_train, y_shuffle_val, y_shuffle_test = split_data(data=shuffled_data_set)

    print (X_shuffle_train.shape)
    print (X_shuffle_val.shape)
    print (X_shuffle_test.shape)

    print (y_shuffle_train.shape)
    print (y_shuffle_val.shape)
    print (y_shuffle_test.shape)

    '''
    EXERCISE 2
    '''
    # fit_curve(X=X_shuffle_train[0], yy=y_shuffle_train[0])

    '''
    EXERCISE 3
    '''
    choose_polynomial(X=X_shuffle_train[0], yy=y_shuffle_train[0])



    # q = [12, 232, 34, 323, 421, 544, 63, 23, 75, 6833, 5324, 453, 34, 31, 797, 313, 1213, 89]
    # q = np.array(q).reshape(6,3)
    # for i in range(3):
    #     sh = shuffle_matrix(q)
    #     print (sh)


    # # Reset random number generator and data provider states on each run
    # # to ensure reproducibility of results
    # rng.seed(seed)
    # train_data.reset()
    # valid_data.reset()



# Split data into training, validation and test sets    (UNUSED)
def split_data(data):
    p = math.floor(len(data) / 20 * 3)  # 15% (validation set / testing set)
    r = len(data) - 2 * p               # 80% (training set)

    training_data = (data[:r])          # training_data
    validation_data = (data[r:r+p])     # validation_data
    testing_data = (data[r+p:r+p+p])    # testing_data

    X_shuffle_train = training_data[:,:-1]
    y_shuffle_train = training_data[:,-1]

    X_shuffle_val = validation_data[:,:-1]
    y_shuffle_val = validation_data[:,-1]

    X_shuffle_test = testing_data[:,:-1]
    y_shuffle_test = testing_data[:,-1]

    return X_shuffle_train, X_shuffle_val, X_shuffle_test, y_shuffle_train, y_shuffle_val, y_shuffle_test


def shuffle_matrix(mat):
    np.random.seed(seed)
    shuffled_mat = np.random.permutation(mat)

    # print (mat)
    # print (shuffled_mat)
    return shuffled_mat


def fit_and_plot(phi_fn, type, X, yy, last_point, draw_type, D):
    w_fit = np.linalg.lstsq(phi_fn(X), yy, rcond=0)[0]   # (D+1,)
    X_grid = np.arange(start=0, stop=1, step=0.05).reshape(-1, D)   # (N, 1)
    f_grid = np.dot(phi_fn(X_grid), w_fit)  # (N,)


    est_value = last_point[0] * w_fit[1] + w_fit[0]
    print ("Estimated value for", type, ":", est_value[0])

    # plt.clf()
    plt.plot(X, yy, 'r.')
    plt.plot(last_point[0], last_point[1], 'b.')
    plt.plot(X_grid, f_grid, draw_type)
    # plt.show()

def phi_linear(X_in):
    return np.insert(X_in, 0, 1, axis=-1)

def phi_quartic(X_in):
    return np.concatenate([np.ones(X_in.shape), X_in, X_in**2, X_in**3, X_in**4], axis=-1)

'''
EXERCISE 2
'''
def fit_curve(X, yy):
    X = X.reshape(20,1)
    yy = yy.reshape(1,1)
    D = X.shape[1]                      # N = 1, D = 20

    # y_data = X                                                  # (N, D) - (20, 1)
    t = np.arange(start=0, stop=1, step=0.05).reshape(-1, D)    # (N, D) - (20, 1)
    # print ("T: ", t)
    # x_data = np.insert(t, 0, np.ones(t.shape[0]), axis=1)       # (N, D+1) - (20, 2)
    #
    # # print ("X: ", X.shape)
    # # print ("Y: ", yy.shape)
    # w_fit = np.linalg.lstsq(x_data, y_data)[0]   # (D,)
    # # X_grid = np.arange(start=0, stop=1, step=0.05).reshape(-1, D) # (N, D)
    # # f_grid = np.dot(X_grid, w_fit)  # (N,)
    # plt.clf()
    # plt.plot(t, y_data, 'r.')
    # # plt.ylim(-0.210540, -0.210530)
    # plt.show()

    print ("Expected value: ", yy[0][0])

    print ("USING ALL DATA")
    plt.clf()
    fit_and_plot(phi_fn=phi_linear, type='linear', X=t, yy=X, last_point=(1, yy[0]), draw_type='g-', D=D)
    fit_and_plot(phi_fn=phi_quartic, type='quartic', X=t, yy=X, last_point=(1, yy[0]), draw_type='y-', D=D)
    plt.show()

    print ("USING JUST THE LAST TWO POINTS")
    plt.clf()
    fit_and_plot(phi_fn=phi_linear, type='linear', X=t[-2:], yy=X[-2:], last_point=(1, yy[0]), draw_type='g-', D=D)
    fit_and_plot(phi_fn=phi_quartic, type='quartic', X=t[-2:], yy=X[-2:], last_point=(1, yy[0]), draw_type='y-', D=D)
    plt.show()


def Phi(C, K, t):
    def phi(X_in, C, K):
        arr = np.ones((C, 1))
        for i in range(K):
            arr = np.hstack((arr, X_in**(i+1)))
        return arr

    res = phi(t, C, K)
    print (res)
    return res


'''
EXERCISE 3
'''
def choose_polynomial(X, yy):
    X = X.reshape(20,1)
    D = X.shape[1]
    t = np.arange(start=0, stop=1, step=0.05).reshape(-1, D)
    Phi(t, 3, 4)





if __name__ == '__main__':
    process_data()
