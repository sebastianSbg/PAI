import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from sklearn.gaussian_process.kernels import Matern, ConstantKernel, Sum, Product
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

domain = np.array([[0, 5]])


""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here

        self.v_min = 1.2
        noise_obs_f = 0.15
        noise_obs_v = 0.0001
        
        # Priors for f and v
        f_variance = np.sqrt(0.5)
        self.kernel_f = Product(Matern(length_scale=0.5, length_scale_bounds=[1e-5,1e5], nu=2.5), ConstantKernel(f_variance))
        self.f = GPR(kernel=self.kernel_f, alpha=noise_obs_f**2, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)

        v_variance = np.sqrt(np.sqrt(2))
        v_mean = 1.5
        self.kernel_v = Sum(ConstantKernel(v_mean, constant_value_bounds="fixed"), Product(Matern(length_scale=0.5, length_scale_bounds=[1e-5,1e5], nu=2.5), ConstantKernel(v_variance)))
        self.v = GPR(kernel=self.kernel_v, alpha=noise_obs_v**2, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)

        # Prealocate data_array
        self.data_points = np.zeros([1,3])
        self.data_points_aux = np.zeros([1,3])

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
        return x_recommended
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
        '''
        print(x_values)
        print()
        print(f_values)
        print()
        print(x0_values)
        print(objective(x_values[14]))
        print()
        
        print(x0_values)
        print(self.f.predict(x0_values))
        print(self.v.predict(x0_values))
        '''

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
        k = 10

        # Importance of good accuracy
        alpha = 1
        tau = 500
        beta = 100

        # Trying LCB acquisition function first
        af_value = f_x[0] + k*f_x[1] - 1./(1+np.exp(-beta*(self.v_min - v_x[0])))
        #print(af_value)
        af_value = np.reshape(af_value,[1])

        #print(f_x)
        #print(v_x)
        #print(af_value)

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

        self.f.fit(np.atleast_2d(data_array[:,0]), np.atleast_2d(data_array[:,1]))
        self.v.fit(np.atleast_2d(data_array[:,0]), np.atleast_2d(data_array[:,2]))

        '''
        print(self.f.kernel_.theta, self.v.kernel_.theta)
        print(self.f.kernel.theta, self.v.kernel.theta)
        print()
        '''

        data_array = np.append(x,np.atleast_2d(self.f.predict(x)),axis=1)
        data_array = np.append(data_array,np.atleast_2d(self.v.predict(x)),axis=1)
        self.data_points_aux = np.vstack((self.data_points_aux, data_array))

        #print(self.f.predict(np.atleast_2d(2.7), return_std=True))
        #print(f_aux(2.7))
        

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
            opt_idx = np.argmax(suitable_points[:,1])
        except:
            opt_idx = np.argmax(self.data_points[:,1])
            suitable_points = self.data_points


        print("Real Points:\n", self.data_points)
        print("\nEstimated Points:\n", self.data_points_aux)
        print("Selected Point:")
        print(suitable_points[opt_idx])
        print(self.acquisition_function(suitable_points[opt_idx,0]), self.f.predict(np.atleast_2d(suitable_points[opt_idx,0]), return_std=True))

        return suitable_points[opt_idx, 0]

        # TODO: finish code

""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])

f_variance_hidden = np.sqrt(0.5)
kernel_f_hidden = Product(Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5), ConstantKernel(f_variance_hidden))

def f_aux(x):
    """Dummy objective"""
    return kernel_f_hidden(np.atleast_2d(x)) + np.random.normal(loc=0.0, scale=0.15)

v_variance_hidden = np.sqrt(np.sqrt(2))
v_mean = 1.5
kernel_v_hidden = Sum(ConstantKernel(v_mean,constant_value_bounds="fixed"), Product(Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5),ConstantKernel(v_variance_hidden)))
def v_aux(x):
    """Dummy speed"""
    return kernel_v_hidden(np.atleast_2d(x)) + np.random.normal(loc=0.0, scale=0.0001)


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