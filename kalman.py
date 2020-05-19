import numpy as np
import datetime as dt
import os


class Kalman:

    def __init__(self, measures, target, dates, steps, q, r, result_dir):
        time = dt.datetime.now()
        self.result_path = os.path.join(result_dir, '%s-kf-result.txt' %
                                        (time.strftime('%d%m%Y-%H%M%S')))
        self.result_fig_path = os.path.join(result_dir, '%s-kf-result.png' %
                                            (time.strftime('%d%m%Y-%H%M%S')))
        self.steps = steps
        self.measures, self.target, self.dates = measures, target, dates
        sz = (self.steps,)  # size of array
        # allocate space for arrays
        self.x_posterior = np.zeros(sz)  # a posteri estimate of x
        self.P_posterior = np.zeros(sz)  # a posteri error covariance
        self.x_prior = np.zeros(sz)  # a priori estimate of x
        self.P_prior = np.zeros(sz)  # a priori error covariance
        self.K = np.zeros(sz)  # gain or blending factor
        self.abs_errors = np.zeros(sz)
        self.squared_errors = np.zeros(sz)

        self.Q = q  # process variance
        self.R = r  # estimate of measurement variance

        # initial values
        self.x_posterior[0] = 0.0
        self.P_posterior[0] = 1.0

    def run_kalman_filter(self):
        for k in range(1, self.steps):
            # time update
            self.x_prior[k] = self.x_posterior[k - 1]
            self.P_prior[k] = self.P_posterior[k - 1] + self.Q

            # measurement update
            self.K[k] = self.P_prior[k] / (self.P_prior[k] + self.R)
            self.x_posterior[k] = self.x_prior[k] + self.K[k] * (self.measures[k] - self.x_prior[k])
            self.P_posterior[k] = (1 - self.K[k]) * self.P_prior[k]
            self.squared_errors[k] = (self.x_posterior[k] - self.target[k]) ** 2
            self.abs_errors[k] = np.abs(self.x_posterior[k] - self.target[k])

        mse = self.squared_errors.sum() / self.steps
        mae = self.abs_errors.sum() / self.steps

        return mse, mae
