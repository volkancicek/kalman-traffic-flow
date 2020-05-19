import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os


class Kalman:

    def __init__(self, data, iter, Q, R):
        self.iter = iter
        self.data = data
        self.measures, self.target, self.dates = self.data.get_test_data()
        sz = (self.iter,)  # size of array
        # allocate space for arrays
        self.x_posterior = np.zeros(sz)  # a posteri estimate of x
        self.P_posterior = np.zeros(sz)  # a posteri error covariance
        self.x_prior = np.zeros(sz)  # a priori estimate of x
        self.P_prior = np.zeros(sz)  # a priori error covariance
        self.K = np.zeros(sz)  # gain or blending factor
        self.abs_errors = np.zeros(sz)
        self.squared_errors = np.zeros(sz)

        self.Q = Q  # process variance
        self.R = R  # estimate of measurement variance

        # initial values
        self.x_posterior[0] = 0.0
        self.P_posterior[0] = 1.0

    def run_kalman_filter(self, result_dir):
        for k in range(1, self.iter):
            # time update
            self.x_prior[k] = self.x_posterior[k - 1]
            self.P_prior[k] = self.P_posterior[k - 1] + self.Q

            # measurement update
            self.K[k] = self.P_prior[k] / (self.P_prior[k] + self.R)
            self.x_posterior[k] = self.x_prior[k] + self.K[k] * (self.measures[k] - self.x_prior[k])
            self.P_posterior[k] = (1 - self.K[k]) * self.P_prior[k]
            self.squared_errors[k] = (self.x_posterior[k] - self.target[k]) ** 2
            self.abs_errors[k] = np.abs(self.x_posterior[k] - self.target[k])

        mse = self.squared_errors.sum() / self.iter
        mae = self.abs_errors.sum() / self.iter

        result_path = os.path.join(result_dir, '%s-kalman-result.txt' %
                                   (dt.datetime.now().strftime('%d%m%Y-%H%M%S')))
        return mse, mae, result_path

    def plot_results(self, normalized):
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.figure()
        if normalized:
            self.measures = self.data.denormalize_measures(self.measures)
            self.target = self.data.denormalize_target(self.target)
            self.x_posterior = self.data.denormalize_target(self.x_posterior)

        plt.plot(self.measures, 'k+', label='noisy measurements')
        plt.plot(self.x_posterior, 'b-', label='prediction')
        plt.plot(self.target, color='g', label='true value')
        plt.legend()
        plt.title('Prediction vs. iteration', fontweight='bold')
        plt.xlabel('dates')

        x_label_count = int(len(self.dates) / 70)
        xi = list(range(x_label_count))
        xi = [i * 70 for i in xi]
        x_axes = [self.dates[i] for i in xi]
        plt.xticks(xi, x_axes)
        plt.ylabel('volume per hour')
        plt.show()

        plt.figure()
        valid_iter = range(1, self.iter)  # Pminus not valid at step 0
        plt.plot(valid_iter, self.P_prior[valid_iter], label='a priori error estimate')
        plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('$(VolPerHour)^2$')
        plt.setp(plt.gca(), 'ylim', [0, .01])
        plt.show()


