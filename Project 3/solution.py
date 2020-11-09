import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from sklearn.gaussian_process.kernels import Matern, ConstantKernel, Sum, Product, WhiteKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.cluster import KMeans
import time

domain = np.array([[0, 5]])


""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        #self.File_object = open(r"/Users/adrialopezescoriza/Documents/UNI/GRAU/4A-ETH/PAI/PROJECTS/PAI/Project 3/results.txt","w+")

        self.v_min = 1.2
        noise_obs_f = 0.15
        noise_obs_v = 0.0001**2
        
        # Priors for f and v
        f_variance = 0.5
        self.kernel_f = Sum(Product(Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5), ConstantKernel(f_variance,constant_value_bounds="fixed")), WhiteKernel(noise_obs_f))
        #self.kernel_f = RBF(1) * ConstantKernel(1)
        self.f = GPR(kernel=self.kernel_f, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)

        v_variance = np.sqrt(2)
        v_mean = 1.5
        self.kernel_v = Sum(Sum(ConstantKernel(v_mean), Product(Matern(length_scale=0.5, length_scale_bounds=[1e-5,1e5], nu=2.5), ConstantKernel(v_variance))), WhiteKernel(noise_obs_v))
        #self.kernel_v = ConstantKernel(v_mean)
        self.v = GPR(kernel=self.kernel_v, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)

        # Prealocate data_array
        self.data_points = np.zeros([1,3])
        self.data_points_aux = np.zeros([1,5])

        # TODO: finish code


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        x_recommended = self.optimize_acquisition_function()
        #x_recommended = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])
        return np.atleast_2d(x_recommended)
        # TODO: finish code


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []
        x0_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])
            x0_values.append(x0)

        ind = np.argmax(f_values)

        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here

        f_x = self.f.predict(np.atleast_2d(x), return_std=True)
        v_x = self.v.predict(np.atleast_2d(x), return_std=True)

        # k as Exploration-Exploitation trade-off
        k = 4

        # Importance of good accuracy
        tau = -8
        beta = 20
        alpha = 1

        # Trying LCB acquisition function first
        #af_value = f_x[0] + k*f_x[1] - tau*1./(1+np.exp(-beta*(self.v_min - v_x[0])))
        #print(af_value)

        #af_value = alpha*f_x[0] + f_x[1] - tau*1./(1+np.exp(-beta*(self.v_min - v_x[0])))
        af_value = alpha*f_x[0] + k*f_x[1] #- tau*1./(1+np.exp(-beta*(self.v_min - v_x[0])))
        if(v_x[0] < self.v_min):
            af_value = -10

        af_value = np.reshape(af_value,[1])

        return af_value

        # TODO: finish code


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here

        # Add data point to data array
        data_array = np.append(x,np.atleast_2d(f),axis=1)
        data_array = np.append(data_array,np.atleast_2d(v),axis=1)
        self.data_points = np.vstack((self.data_points, data_array))

        if(np.sum(self.data_points[0,:]) == 0):
            self.data_points = np.delete(self.data_points, 0, axis=0)
            self.data_points_aux = np.delete(self.data_points_aux, 0, axis=0)

        data_array = np.append(x,np.atleast_2d(self.f.predict(x)),axis=1)
        data_array = np.append(data_array,np.atleast_2d(self.v.predict(x)),axis=1)
        data_array = np.append(data_array,np.atleast_2d(self.f.predict(x,return_std=True)[1]),axis=1)
        data_array = np.append(data_array,np.atleast_2d(self.v.predict(x,return_std=True)[1]),axis=1)
        self.data_points_aux = np.vstack((self.data_points_aux, data_array))

        '''
        Npoints = 10
        if(np.shape(self.data_points)[0] > Npoints):
            data_transformed = KMeans(n_clusters=Npoints, random_state=0).fit(self.data_points[:,0:2]).cluster_centers_
        else:
            data_transformed = self.data_points
        self.f.fit(np.atleast_2d(data_transformed[:,0]).T, np.atleast_2d(data_transformed[:,1]).T)
        '''

        self.f.fit(np.atleast_2d(self.data_points[:,0]).T, np.atleast_2d(self.data_points[:,1]).T)
        self.v.fit(np.atleast_2d(self.data_points[:,0]).T, np.atleast_2d(self.data_points[:,2]).T)

        # TODO: finish code

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        
        # minimum speed recquired
        mask = (self.data_points[:, 2] > self.v_min)
        suitable_points = self.data_points[mask]
        try:
            #h = hello
            opt_idx = np.argmax(suitable_points[:,1])
        except:
            opt_idx = np.argmax(self.data_points[:,1])
            suitable_points = self.data_points


        # Comments to check out how everything is going
        print("Real Points:\n", self.data_points)
        print("\nEstimated Points:\n", self.data_points_aux[:,:])
        print("Selected Point:")
        print(suitable_points[opt_idx])
        print(self.acquisition_function(suitable_points[opt_idx,0]), self.f.predict(np.atleast_2d(suitable_points[opt_idx,0]), return_std=True))
        print("F Regression Score:", self.f.score(np.atleast_2d(self.data_points[:,0]).T, np.atleast_2d(self.data_points[:,1]).T))
        print("V Regression Score:", self.v.score(np.atleast_2d(self.data_points[:,0]).T, np.atleast_2d(self.data_points[:,2]).T))

        
        #self.File_object.write("Real Points:\n" + str(self.data_points) + "\nEstimated Points:\n" +  str(self.data_points_aux[:,:]) + "\nSelected Point:" + str(suitable_points[opt_idx]))
        #File_object.close()
        #time.sleep(5000)
        return suitable_points[opt_idx, 0]

        # TODO: finish code

""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])

f_variance_hidden = 0.5
kernel_f_hidden = Matern(length_scale=4.5, length_scale_bounds="fixed", nu=2.5) + WhiteKernel(f_variance_hidden)

def f_aux(x):
    """Dummy objective"""
    return 2-(3.7-x)**2 + np.random.normal(loc=0.0, scale=0.15)

v_variance_hidden = np.sqrt(np.sqrt(2))
v_mean = 1.5
kernel_v_hidden = Sum(ConstantKernel(v_mean,constant_value_bounds="fixed"), Product(Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5),ConstantKernel(v_variance_hidden)))
def v_aux(x):
    """Dummy speed"""
    return 5-x + np.random.normal(loc=0.0, scale=0.0001)


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f_aux(x)
        cost_val = v_aux(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v_aux(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f_aux(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f_aux(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()