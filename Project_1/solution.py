import numpy as np

import scipy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel, DotProduct, ExpSineSquared, RationalQuadratic, Sum
from sklearn.kernel_approximation import Nystroem

## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model():

    def __init__(self):
        """
            TODO: enter your code here
        """
        self.kernel = ConstantKernel() + ConstantKernel()*RBF()
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=0,random_state=0)

    def predict(self, test_x):
        """
            TODO: enter your code here
        """

        y = self.model.predict(test_x)
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        data_xy = np.column_stack((train_x,train_y))
        rng = np.random.default_rng()
        approx = 'Random'

        # Nyostream approximation (not implemented yet)
        if(approx == 'Nystroem'):
            feature_map_nystroem = Nystroem(kernel='rbf',gamma=.2,random_state=1,n_components=300)
            data_transformed = feature_map_nystroem.fit_transform(train_x)

        # Select random samples from dataset
        if(approx == 'Random'):
            n = 10
            data_transformed = rng.choice(data_xy,size=n, axis=0, replace=False)

        
        # Clusterize data into 
        if(approx == 'Clusters'):
            n_clusters = 20
            dist_thresehold = 0.07
            cluster_centers = rng.choice(data_xy,size=n_clusters, axis=0, replace=False)
            data_transformed = np.zeros((1,3))
            for i in range(n_clusters):
                cluster_center = cluster_centers[i,0:3]
                for point in data_xy:
                    dist = np.linalg.norm(cluster_center-point)
                    if(dist < dist_thresehold):
                        point_app = np.reshape(point,(1,3))
                        if(data_transformed.all()==0):
                            data_transformed = point_app
                            continue

                        data_transformed = np.append(data_transformed,point_app,axis=0)

        # Using entire data set
        if(approx == 'None'):
            data_transformed = data_xy
        
        data_transformed_x = data_transformed[:,0:2]
        data_transformed_y = data_transformed[:,2]        
        
        self.data_x = train_x
        self.data_y = train_y

        self.model.fit(data_transformed_x, data_transformed_y)

        self.model.kernel_.theta = self.optimizer()


    def obj_func(self,hyperparams)->float:

            self.model.kernel_.theta = hyperparams
            prediction = self.model.predict(self.data_x)
            cost = cost_function(self.data_y,prediction)
            self.model.kernel.theta = hyperparams
            return cost

    
    def optimizer(self):
        # * 'obj_func' is the objective function to be minimized, which
        #   takes the hyperparameters theta as parameter and an
        #   optional flag eval_gradient, which determines if the
        #   gradient is returned additionally to the function value
        # * 'initial_theta': the initial value for theta, which can be
        #   used by local optimizers
        # * 'bounds': the bounds on the values of theta
        #....
        # Returned are the best found hyperparameters theta and
        # the corresponding value of the target function.

        initial_theta = self.model.kernel_.theta
        optimalResult = scipy.optimize.minimize(self.obj_func, initial_theta, method='BFGS')
        theta_opt = optimalResult.x
        return theta_opt

def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(train_x)
    print(cost_function(prediction,train_y))

if __name__ == "__main__":
    main()
    print("Completed sucessfully")
