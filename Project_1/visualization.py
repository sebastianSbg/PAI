import numpy as np
import matplotlib.pyplot as plt

def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')    

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    #rng = np.random.default_rng()
    #tr = rng.choice(data_xy,size=n, axis=0, replace=False)            

    # plotting data
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(train_x[:,0], train_x[:,1], train_y, c=train_y, cmap='Greens');    
    ax.scatter3D(test_x[:,0], test_x[:,1], np.zeros(test_x[:,0].shape), c='r')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.show()

if __name__ == "__main__":
    main()
    print("Completed sucessfully")