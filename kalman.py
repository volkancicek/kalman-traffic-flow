import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

plt.rcParams['figure.figsize'] = (10, 8)

# initial parameters
n_iter = 288
sz = (n_iter,)  # size of array
configs = json.load(open('config.json', 'r'))
df = pd.read_csv(os.path.join(configs['data']['base_path'], configs['data']['approach_1']['data_file_name']))
target_col = configs['data']['target_column']
measure_columns = configs['data']['measures']
dates = df.get("date").values[1151:1439]
measures = df.get(measure_columns).values[1150:1438]
target = df.get(target_col).values[1151:1439]

# allocate space for arrays
xhat = np.zeros(sz)  # a posteri estimate of x
P = np.zeros(sz)  # a posteri error estimate
xhatminus = np.zeros(sz)  # a priori estimate of x
Pminus = np.zeros(sz)  # a priori error estimate
K = np.zeros(sz)  # gain or blending factor
sq_errors = np.zeros(sz)

Q = 0.1 ** 5  # process variance
R = 0.1 ** 2  # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
P[0] = 1.0

for k in range(1, n_iter):
    # time update
    xhatminus[k] = xhat[k - 1]
    Pminus[k] = P[k - 1] + Q

    # measurement update
    K[k] = Pminus[k] / (Pminus[k] + R)
    xhat[k] = xhatminus[k] + K[k] * (measures[k] - xhatminus[k])
    P[k] = (1 - K[k]) * Pminus[k]
    sq_errors[k] = (xhat[k] - target[k]) ** 2

plt.figure()
plt.plot(measures, 'k+', label='noisy measurements')
plt.plot(xhat, 'b-', label='a posteri estimate')
plt.plot(target, color='g', label='truth value')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('dates')

x_label_count = int(len(dates) / 70)
xi = list(range(x_label_count))
xi = [i * 70 for i in xi]
x_axes = [dates[i] for i in xi]
plt.xticks(xi, x_axes)
plt.ylabel('volume per hour')
plt.show()

plt.figure()
valid_iter = range(1, n_iter)  # Pminus not valid at step 0
plt.plot(valid_iter, Pminus[valid_iter], label='a priori error estimate')
plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('$(VolPerHour)^2$')
plt.setp(plt.gca(), 'ylim', [0, .01])
plt.show()

print("\n rmse:")
print(np.sqrt(sq_errors.sum() / n_iter))
