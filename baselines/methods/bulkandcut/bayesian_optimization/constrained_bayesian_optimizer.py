import warnings
import csv
from typing import List

import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize

from baselines.methods.bulkandcut import global_seed, rng


class ConstrainedBayesianOptimizer():
    """
    I minimize stuff using an arbitrary subset of the search dimensions.
    """
    def __init__(self, par_bounds: List[dict]):
        self.surrogate_model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=global_seed,
            )
        self.par_bounds = par_bounds
        self.par_names = list(par_bounds.keys())  # To have a fixed reference order

        self.par_values = []
        self.par_targets = []

    @property
    def incumbent(self):
        return self.par_values[np.argmin(self.par_targets)]

    @property
    def n_pars(self):
        return len(self.par_bounds)

    def register_target(self, par_values: dict, target: float):
        # TODO check bounds of par_values, if out ouf bounds, raise warning
        self.par_targets.append(target)
        self.par_values.append([par_values[pname] for pname in self.par_names])

    def next_pars(self, dictated_pars: dict):
        # check validity of dictated pars:
        for dpar_k, dpar_v in dictated_pars.items():
            if dpar_v < self.par_bounds[dpar_k][0] or dpar_v > self.par_bounds[dpar_k][1]:
                print(f"WARNING: Dictaded parameter {dpar_k} is out of bounds. It has value "
                      f"{dpar_v}, but it should be between {self.par_bounds[dpar_k]}")

        lowb, highb = self._get_constrained_bounds(dpars=dictated_pars)

        if len(self.par_targets) < 2:
            # Return a random point if we've seen less than two points.
            suggestion = rng.uniform(low=lowb, high=highb)
        else:
            # Otherwise first we fit the Gaussian Process
            print('Values:', self.par_values)
            print('Targets:', self.par_targets)
            with warnings.catch_warnings():  # TODO: can I get rid of these warnings some other way?
                warnings.simplefilter("ignore")
                self.surrogate_model.fit(
                    X=np.array(self.par_values),
                    y=self.par_targets,
                    )
            # Then we return the LCB minimizer
            suggestion = self._minimize_lcb(lowb, highb)

        # Wrap the suggestion in a dictionary:
        suggestion = {pname: suggestion[n] for n, pname in enumerate(self.par_names)}
        return suggestion

    def _get_constrained_bounds(self, dpars: dict):
        low_bound, high_bound = [], []
        for pname in self.par_names:
            if pname in dpars:
                low_bound.append(dpars[pname])
                high_bound.append(dpars[pname])
            else:
                low_bound.append(self.par_bounds[pname][0])
                high_bound.append(self.par_bounds[pname][1])

        return np.array(low_bound), np.array(high_bound)

    def _minimize_lcb(self,
                      lowb: "np.array",
                      highb: "np.array",
                      n_random: int = 10000,
                      n_solver: int = 10,
                      ):
        """
        A function to find the minimum of the acquisition function It uses a combination of random
        sampling (cheap) and the 'L-BFGS-B' optimization method. First by sampling `n_random` points
        at random, and then running L-BFGS-B for `n_solver` random starting points.

        This function was inspired on (a.k.a. plagiarized from)
        https://github.com/fmfn/BayesianOptimization
        """

        def lcb(x, alpha=2.5):
            """ LCB: lower confidence bound """
            x = x.reshape(1, -1) if x.ndim == 1 else x
            mean, std = self.surrogate_model.predict(X=x, return_std=True)
            return mean - alpha * std

        # Warm up with random points
        x_guesses = rng.uniform(low=lowb, high=highb, size=(n_random, self.n_pars))
        ys = lcb(x=x_guesses)
        best_x = x_guesses[ys.argmin()]
        min_lcb = ys.min()

        # Then use the scipy minimizer solver
        x_guesses = rng.uniform(low=lowb, high=highb, size=(n_solver, self.n_pars))
        x_guesses = np.vstack((best_x, x_guesses))
        scikit_bounds = np.vstack((lowb, highb)).T
        for x0 in x_guesses:
            with warnings.catch_warnings():  # TODO: can I get rid of these warnings some other way?
                warnings.simplefilter("ignore")
                res = minimize(
                    fun=lcb,
                    x0=x0.reshape(1, -1),
                    bounds=scikit_bounds,
                    method="L-BFGS-B",
                    )
            if not res.success:
                continue

            # Store it if better than previous minimum.
            if res.fun[0] < min_lcb:
                best_x = res.x
                min_lcb = res.fun[0]

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        return np.clip(best_x, lowb, highb)

    def save_csv(self, csv_path: str):
        # Write configurations and their respective targets on a csv file
        fieldnames = ["order", "target"] + self.par_names
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(self.par_targets)):
                new_row = {
                    "order": i,
                    "target": self.par_targets[i],
                }
                for p, pname in enumerate(self.par_names):
                    new_row[pname] = self.par_values[i][p]
                writer.writerow(new_row)
