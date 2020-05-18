import numpy as np
import matplotlib.pyplot as plt


class Kalman:
    plt.rcParams['figure.figsize'] = (10, 8)

    def __init__(self, measures, target, dates, n_iter):
        self.iter = n_iter
        self.measures = measures
        self.target = target
        self.dates = dates
        sz = (n_iter,)  # size of array
        # allocate space for arrays
        self.x_posterior = np.zeros(sz)  # a posteri estimate of x
        self.P = np.zeros(sz)  # a posteri error estimate
        self.x_prior = np.zeros(sz)  # a priori estimate of x
        self.Pminus = np.zeros(sz)  # a priori error estimate
        self.K = np.zeros(sz)  # gain or blending factor
        self.squared_errors = np.zeros(sz)

        self.Q = 0.1 ** 2  # process variance
        self.R = 0.1 ** 1  # estimate of measurement variance, change to see effect

        # intial values
        self.x_posterior[0] = 0.0
        self.P[0] = 1.0

    def run_kalman(self):
        for k in range(1, self.iter):
            # time update
            self.x_prior[k] = self.x_posterior[k - 1]
            self.Pminus[k] = self.P[k - 1] + self.Q

            # measurement update
            self.K[k] = self.Pminus[k] / (self.Pminus[k] + self.R)
            self.x_posterior[k] = self.x_prior[k] + self.K[k] * (self.measures[k] - self.x_prior[k])
            self.P[k] = (1 - self.K[k]) * self.Pminus[k]
            self.squared_errors[k] = (self.x_posterior[k] - self.target[k]) ** 2

    def plot_results(self):
        plt.figure()
        plt.plot(self.measures, 'k+', label='noisy measurements')
        plt.plot(self.x_posterior, 'b-', label='a posteri estimate')
        plt.plot(self.target, color='g', label='true value')
        plt.legend()
        plt.title('Prediction vs. iteration step', fontweight='bold')
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
        plt.plot(valid_iter, self.Pminus[valid_iter], label='a priori error estimate')
        plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('$(VolPerHour)^2$')
        plt.setp(plt.gca(), 'ylim', [0, .01])
        plt.show()

        print("\n mse:")
        print(self.squared_errors.sum() / self.iter)
