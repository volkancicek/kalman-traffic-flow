import json
import os

from data.process_data import ProcessData
from kalman import Kalman
from plot_results import plot_results
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
    """  """

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
    uk = UnscentedKalman(measures, target, dates, steps, float(q), float(r), results_dir)
    mse, mae = uk.run_unscented_kalman()
    save_results(mse, mae, uk.result_path, str(q), str(r))
    return uk


def run_kalman_filter(measures, target, dates, steps, q, r, results_dir):
    k = Kalman(measures, target, dates, steps, float(q), float(r), results_dir)
    mse, mae = k.run_kalman_filter()
    save_results(mse, mae, k.result_path, str(q), str(r))
    return k


def save_results(mse, mae, result_path, q, r):
    file = open(result_path, "w")
    file.write("RESULTS - mse: " + str(mse) + ", mae: " + str(mae) + ",\n"
                                                                     "PARAMETERS - Q: " + str(q) + ", R: " + str(r))
    file.close()
    print("\n filter results are saved to : \n" + str(result_path))
    print("\n mse:")
    print(mse)
    print("\n mae:")
    print(mae)


if __name__ == '__main__':
    main()
