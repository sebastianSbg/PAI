import numpy as np

import scipy

from sklearn.kernel_approximation import RBFSampler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel, DotProduct, ExpSineSquared, RationalQuadratic, Sum
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, Ridge

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
            
        #self.kernel_ = ConstantKernel() + ConstantKernel()*RBF()
        #self.model = GaussianProcessRegressor(kernel=self.kernel_, n_restarts_optimizer=0,random_state=0)
        self.model =  BayesianRidge();

    def predict(self, test_x):       
        # predict with model at test point test_x 

        y = self.model.predict(test_x)
        return y

    def load_and_tranform_data(self, approx, train_x, train_y):        
        data_xy = np.column_stack((train_x,train_y))
        rng = np.random.default_rng()

        # truncate data set to cut off all points where x1 < -0.5
        #data_xy = data_xy[data_xy[:,0] >= -0.5, :]        

        # Nyostream approximation (not implemented yet)
        if(approx == 'Nystroem'):
            feature_map_nystroem = Nystroem(kernel='rbf',n_components=2)                        
            X_features = feature_map_nystroem.fit_transform(data_xy[:,0:2])
            data_transformed = np.column_stack((X_features, data_xy[:,2]))

        # Select random samples from dataset
        if(approx == 'Random'):
            n = 100
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

        #  RBFSampler - Approximates feature map of an RBF kernel by Monte Carlo approximation of its Fourier transform
        if(approx == 'RBFSampler'):
            rbf_feature = RBFSampler()
            X_features = rbf_feature.fit_transform(data_xy[:,0:2])
            data_transformed = np.column_stack((X_features, data_xy[:,2]))


        # Using entire data set
        if(approx == 'None'):
            data_transformed = data_xy

        print("Size of transformed data: " + str(data_transformed.shape))    
        return data_transformed

    def fit_model(self, train_x, train_y):

        # load and transform data according to different methods
        data = self.load_and_tranform_data('Nystroem', train_x, train_y)
        self.data_x = data[:,0:2]
        self.data_y = data[:,2]

        # initliaze model with kernel (necessary before actually doing optimization with our custom cost function)        
        self.model.fit(self.data_x, self.data_y)                

        # actually optimize hyperparameters according to custom cost function
        

        self.model.coef_ =  self.optimizer()              

        # CROSS VALIDATION
        # Trying different initializartions via cross validation
        n_restarts= 100
        rng = np.random.default_rng()
        cost_cv = np.zeros((n_restarts,1))
        theta = np.zeros((n_restarts, self.model.coef_.shape[0]))
        for i in range(n_restarts):

            # Random parametre initiallization
            random_initialization = np.random.randn(1,self.model.coef_.shape[0])[0,:]
            print('\n',random_initialization)
            self.model.coef_ = random_initialization
            self.model.coef_ = self.optimizer()
            print(self.model.coef_)

            # Select new cross validation data set
            d = rng.choice(data,size=200, axis=0, replace=False)
            x = d[:,0:2]
            y = d[:,2:3]
            cost_cv[i] = cost_function(y,self.model.predict(x))
            theta[i,:] = self.model.coef_
            print(cost_cv[i])
        
        self.model.coef_ = theta[np.argmin(cost_cv,axis=1)[0],:]        

    def obj_func(self,hyperparams)->float:
        self.model.coef_ = hyperparams
        prediction = self.model.predict(self.data_x)
        cost = cost_function(self.data_y,prediction)
        self.model.coef_ = hyperparams

        return cost

    def optimizer(self):
        initial_theta = self.model.coef_                
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
    print("Cost on train_x data: " + str(cost_function(train_y, prediction)))

if __name__ == "__main__":
    main()
    print("Completed sucessfully")
