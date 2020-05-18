import numpy as np
import pandas as pd
import os


class ProcessData:
    """A class for loading and transforming data for RNN models"""

    def __init__(self, configs):
        self.configs = configs
        df = pd.read_csv(os.path.join(configs['data']['base_path'], configs['data']['approach_1']['data_file_name']))
        self.split = int(len(df) * configs['data']['split'])
        self.target_col = configs['data']['target_column']
        self.measure_columns = configs['data']['measures']
        self.dates = df.get("date").values[:]
        self.measures = df.get(self.measure_columns).values[:]
        self.target = df.get(self.target_col).values[:]
        self.measures_mean = self.measures[:self.split].mean(axis=0)
        self.measures_std = self.measures[:self.split].std(axis=0)
        self.target_mean = self.target[:self.split].mean(axis=0)
        self.target_std = self.target[:self.split].std(axis=0)

    def get_test_data(self):
        return np.array(self.measures[self.split:]), np.array(self.target[self.split:]), self.dates[self.split:]

    def normalize_data(self):
        self.measures = (self.measures[:] - self.measures_mean) / self.measures_std
        self.target = (self.target[:] - self.target_mean) / self.target_std

    def denormalize_target(self, t):
        return (t * self.target_std) + self.target_mean

    def denormalize_measures(self, m):
        return (m * self.measures_std) + self.measures_mean
