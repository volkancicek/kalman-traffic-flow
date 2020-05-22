from data.prepare_data import PrepareData
import json


def main():
    configs = json.load(open('../config.json', 'r'))
    features_csv_app1 = configs['data']['approach_1']['feature_csv']
    labels_rpt_path = configs['data']['labels_path']
    # start date : 20/12/2019 00:00:00
    start_timestamp = configs['data']['approach_1']['start_timestamp']
    # end date: 17/03/2020 06:30:30
    data_size = 25422
    # aggregate per 5 min
    aggregate_time = 300000
    data = PrepareData(start_timestamp, data_size, features_csv_app1, labels_rpt_path, aggregate_time)
    data.save_averaged_data_to_csv('approach_1_avg.csv')
    data.save_split_data_to_csv('approach_1_split.csv')


if __name__ == '__main__':
    main()
