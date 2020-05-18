import json
import os

from data.process_data import ProcessData
from kalman import Kalman


def main():
    configs = json.load(open('config.json', 'r'))
    data = ProcessData(configs)
    results_dir = configs['results']
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    if configs['data']['normalise']:
        data.normalize_data()

    iter = 288
    k = Kalman(data, iter)
    # run filter for 11.03.2020
    k.measures = k.measures[425:713]
    k.target = k.target[426:714]
    k.dates = k.dates[426:714]
    k.run_kalman(results_dir)
    k.plot_results(configs['data']['normalise'])


if __name__ == '__main__':
    main()
