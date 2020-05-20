import os
import datetime as dt
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter
import numpy as np
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
from helper import get_trend


class UnscentedKalman:

    def __init__(self, measures, target, dates, steps, q, r, result_dir):
        time = dt.datetime.now()
        self.result_path = os.path.join(result_dir, '%s-ukf-result.txt' %
                                        (time.strftime('%d%m%Y-%H%M%S')))
        self.result_fig_path = os.path.join(result_dir, '%s-ukf-result.png' %
                                            (time.strftime('%d%m%Y-%H%M%S')))
        # self.sigmas = JulierSigmaPoints(n=1, kappa=1)
        self.sigmas = MerweScaledSigmaPoints(1, alpha=.1, beta=2., kappa=1.)
        self.ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, dt=1., hx=self.hx, fx=self.fx, points=self.sigmas)
        self.ukf.P = 1
        self.ukf.R = r
        self.ukf.Q = q  # .0001  # process variance     # Q_discrete_white_noise(2, dt=1., var=0.03)
        self.steps = steps
        self.measures, self.target, self.dates = measures, target, dates
        sz = (self.steps,)  # size of array
        self.x = np.zeros(sz)  # a posteri estimate of x
        self.abs_errors = np.zeros(sz)
        self.squared_errors = np.zeros(sz)
        self.time_point = 0

    def fx(self, x, dt):
        x_out = np.empty_like(x)
        x_out[0] = x[0]
        self.time_point += 1
        return x_out

    def hx(self, x):
        return x * get_trend(self.time_point)

    def run_unscented_kalman(self):
        for k in range(self.steps):
            self.ukf.predict()
            self.ukf.update(self.measures[k])
            self.x[k] = self.ukf.x[0]
            self.squared_errors[k] = (self.ukf.x[0] - self.target[k]) ** 2
            self.abs_errors[k] = np.abs(self.ukf.x[0] - self.target[k])

        mse = self.squared_errors.sum() / self.steps
        mae = self.abs_errors.sum() / self.steps
        print("max-min filter:")
        print(str(self.x.max()))
        print(str(self.x.min()))
        print("max-min target:")
        print(str(self.target.max()))
        print(str(self.target.min()))

        return mse, mae
