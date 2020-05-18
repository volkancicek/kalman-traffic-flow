import json

from data.process_data import ProcessData
from kalman import Kalman


def main():

    configs = json.load(open('config.json', 'r'))
    data = ProcessData(configs)

    if configs['data']['normalise']:
        data.normalize_data()

    measures, target, dates = data.get_test_data()
    n_iter = 288
    k = Kalman(measures[425:713], target[426:714], dates[426:714], n_iter)
    k.run_kalman()
    k.plot_results()


if __name__ == '__main__':
    main()
