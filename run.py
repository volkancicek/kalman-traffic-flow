import json
import os
import numpy as np

from data.process_data import ProcessData
from kalman import Kalman
from helper import plot_results
from unscented_kalman import UnscentedKalman


def main():
    configs = json.load(open('config.json', 'r'))
    data = ProcessData(configs)
    results_dir = configs['results']
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    normalize = configs['data']['normalise']
    if normalize:
        data.normalize_data()

    measures, target, dates = data.get_test_data()
    """ run filters for the  Selected day (11.03.2020) """
    measures = measures[425:713]
    target = target[426:714]
    dates = dates[426:714]

    target_abs_err, target_sq_err = np.zeros(len(target)), np.zeros(len(target))
    for i in range(1, len(target) - 1):
        target_abs_err[i - 1] = np.abs(target[i] - target[i - 1])
        target_sq_err[i - 1] = (target[i] - target[i - 1]) ** 2

    print("\ntarget mean absolute error (MAE): ")
    print(target_abs_err.sum() / len(target))
    print("target mean squared error (MSE):")
    print(target_sq_err.sum() / (len(target)))
    print("\ntarget max-min values:")
    print(str(target.max()))
    print(str(target.min()))

    kf_q = configs['kalman']['Q']
    kf_r = configs['kalman']['R']
    ukf_q = configs['unscented']['Q']
    ukf_r = configs['unscented']['R']
    steps = 288
    kf = run_kalman_filter(measures, target, dates, steps, kf_q, kf_r, results_dir)
    kf_predictions = kf.x_posterior
    ukf = run_unscented_kalman_filter(measures, target, dates, steps, ukf_q, ukf_r, results_dir)
    ukf_predictions = ukf.x

    """ if measures & target are normalized when applying filters, they should be denormalized before plotting! """
    if normalize:
        measures = data.denormalize_measures(measures)
        target = data.denormalize_target(target)
        kf_predictions = data.denormalize_target(kf_predictions)
        ukf_predictions = data.denormalize_target(ukf_predictions)
    """ """

    plot_results(measures, target, kf_predictions, dates, "Kalman Filter Results", kf.result_fig_path)
    plot_results(measures, target, ukf_predictions, dates, "Unscented Kalman Filter Results", ukf.result_fig_path)


def run_unscented_kalman_filter(measures, target, dates, steps, q, r, results_dir):
    ukf = UnscentedKalman(measures, target, dates, steps, float(q), float(r), results_dir)
    mse, mae = ukf.run_unscented_kalman()
    print("\nUKF results:")
    save_results(mse, mae, ukf.result_path, str(q), str(r))
    print("UKF max-min values:")
    print(str(ukf.x.max()))
    print(str(ukf.x.min()))

    return ukf


def run_kalman_filter(measures, target, dates, steps, q, r, results_dir):
    kf = Kalman(measures, target, dates, steps, float(q), float(r), results_dir)
    mse, mae = kf.run_kalman_filter()
    print("\nKF results:")
    save_results(mse, mae, kf.result_path, str(q), str(r))
    print("KF max-min values:")
    print(str(kf.x_posterior.max()))
    print(str(kf.x_posterior.min()))

    return kf


def save_results(mse, mae, result_path, q, r):
    file = open(result_path, "w")
    file.write("RESULTS - mse: " + str(mse) + ", mae: " + str(mae) + ",\n"
                                                                     "PARAMETERS - Q: " + str(q) + ", R: " + str(r))
    file.close()
    print("filter results are saved to : " + str(result_path))
    print("MSE:")
    print(mse)
    print("MAE:")
    print(mae)


if __name__ == '__main__':
    main()
