import json
import os

from data.process_data import ProcessData
from kalman import Kalman


def main():
    configs = json.load(open('config.json', 'r'))
    data = ProcessData(configs)
    results_dir = configs['results']
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    normalize = configs['data']['normalise']
    if normalize:
        data.normalize_data()
    q = configs['Q']
    r = configs['R']
    steps = 288
    run_kalman_filter(data, steps, q, r, normalize, results_dir)


def run_kalman_filter(data, steps, q, r, normalized, results_dir):
    k = Kalman(data, steps, float(q), float(r))
    # run filter for 11.03.2020
    k.measures = k.measures[425:713]
    k.target = k.target[426:714]
    k.dates = k.dates[426:714]
    mse, mae, result_path, = k.run_kalman_filter(results_dir)
    k.plot_results(normalized)
    save_results(mse, mae, result_path, str(q), str(r))


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
