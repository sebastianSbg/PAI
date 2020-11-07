import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

domain = np.array([[0, 5]])


""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        
        # Priors for f and v
        f_variance = 2.5
        kernel_f = Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5) * ConstantKernel(f_variance)
        self.f = GPR(kernel=kernel_f, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)

        v_variance = np.sqrt(2)
        v_mean = 1.5
        kernel_v = ConstantKernel(v_mean) + Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5) * ConstantKernel(v_variance)
        self.v = GPR(kernel=kernel_v, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)

        # Prealocate data_array
        self.data_points = np.empty([1,3])

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

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

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
        k = 1

        # v takes velocity into account
        tau = 0.2

        # Trying LCB acquisition function first
        af_value = f_x[0] + k*f_x[1] + tau*(v_x[0] + v_x[1])
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
        self.data_points = np.append(self.data_points, data_array, axis=0)

        self.f.fit(np.atleast_2d(data_array[:,0]), np.atleast_2d(data_array[:,1]))
        self.v.fit(np.atleast_2d(data_array[:,0]), np.atleast_2d(data_array[:,2]))

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
        v_min = 1.2
        mask = (self.data_points[:, 2] > v_min)
        opt_idx = np.argmax(self.data_points[:, 1][mask])

        return self.data_points[0, opt_idx]

        # TODO: finish code

""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


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
        obj_val = f(x)
        cost_val = v(x)
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
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()