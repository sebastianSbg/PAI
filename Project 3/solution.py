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
        self.x_vals = np.array([])
        self.f_vals = np.array([])
        self.v_vals = np.array([])

        self.data_points = np.zeros([1,3])
        self.data_points_aux = np.zeros([1,5])


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: optimize giving of next recommendation
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        x_recommended = self.optimize_acquisition_function()
        #x_recommended = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])
        return np.atleast_2d(x_recommended)

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

        # TODO: tune acquisition function
        f_x = self.f.predict(np.atleast_2d(x), return_std=True)
        v_x = self.v.predict(np.atleast_2d(x), return_std=True)

        # k as Exploration-Exploitation trade-off
        k = 0.9

        # Trying LCB acquisition function first
        af_value = f_x[0] + k*f_x[1]
        if(v_x[0] < self.v_min):
            af_value = -10

        af_value = np.reshape(af_value,[1])

        return af_value

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

        self.x_vals = np.append(self.x_vals, np.atleast_2d(x))
        self.f_vals = np.append(self.f_vals, np.atleast_2d(f))
        self.v_vals = np.append(self.v_vals, np.atleast_2d(v))

        # fit surrogate model
        self.f.fit(np.atleast_2d(self.x_vals).T, np.atleast_2d(self.f_vals).T)
        self.v.fit(np.atleast_2d(self.x_vals).T, np.atleast_2d(self.v_vals).T)

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
        mask = self.v_vals > self.v_min

        if (np.all(mask == False)): # if v is nowhere larger than v_min, just use all points
            mask = [True for _ in self.v_vals]

        suitable_points = self.f_vals[mask]
        opt_idx = np.argmax(suitable_points)

        # printouts
        print("test points (x): ", self.x_vals)
        print("selected point: ", self.x_vals[mask][opt_idx], ", f: ", self.f_vals[mask][opt_idx])
        print("acquisition function: ", self.acquisition_function(self.x_vals[mask][opt_idx]))
        print("F regression score: ", self.f.score(np.atleast_2d(self.x_vals).T, np.atleast_2d(self.f_vals).T))
        print("V regression score: ", self.v.score(np.atleast_2d(self.x_vals).T, np.atleast_2d(self.v_vals).T))

        return self.x_vals[mask][opt_idx]



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