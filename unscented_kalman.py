import os
import datetime as dt
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import JulierSigmaPoints


class UnscentedKalman:

    def __init__(self, measures, target, dates, steps, q, result_dir):
        time = dt.datetime.now()
        self.result_path = os.path.join(result_dir, '%s-ukf-result.txt' %
                                        (time.strftime('%d%m%Y-%H%M%S')))
        self.result_fig_path = os.path.join(result_dir, '%s-ukf-result.png' %
                                            (time.strftime('%d%m%Y-%H%M%S')))
        self.sigmas = JulierSigmaPoints(n=1, kappa=1)
        self.ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, dt=1., hx=self.hx, fx=self.fx, points=self.sigmas)
        self.ukf.P *= 10
        self.ukf.R *= .2
        self.ukf.Q = q  # process variance     # Q_discrete_white_noise(2, dt=1., var=0.03)
        # self.ukf.R *= R  # estimate of measurement variance
        self.steps = steps
        self.measures, self.target, self.dates = measures, target, dates
        sz = (self.steps,)  # size of array
        self.x = np.zeros(sz)  # a posteri estimate of x
        self.abs_errors = np.zeros(sz)
        self.squared_errors = np.zeros(sz)

    def fx(self, x, dt):
        x_out = np.empty_like(x)
        x_out[0] = x[0]  # + np.cos(dt * 3.14)  # np.log(1 ** x[0]) * dt + x[0]
        return x_out

    def hx(self, x):
        return x[:1]  # return [x]

    def run_unscented_kalman(self):
        for k in range(self.steps):
            self.ukf.predict()
            self.ukf.update(self.measures[k])
            self.x[k] = self.ukf.x[0]
            self.squared_errors[k] = (self.ukf.x[0] - self.target[k]) ** 2
            self.abs_errors[k] = np.abs(self.ukf.x[0] - self.target[k])
        mse = self.squared_errors.sum() / self.steps
        mae = self.abs_errors.sum() / self.steps

        return mse, mae

    def plot_results(self, normalized):
        plt.figure()
        if normalized:
            self.measures = self.data.denormalize_measures(self.measures)
            self.target = self.data.denormalize_target(self.target)
            self.x = self.data.denormalize_target(self.x)

        plt.plot(self.measures, 'k+', label='noisy measurements')
        plt.plot(self.x, 'b-', label='prediction')
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
