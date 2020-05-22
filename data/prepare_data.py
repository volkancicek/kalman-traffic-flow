import pandas as pd
import numpy as np


class PrepareData:

    def __init__(self, start_timestamp, data_size, fcd_path, detector_data_path, aggregate_time_range):

        fcd_df = pd.read_csv(fcd_path)
        columns = ["date", "timestamp", "volume_per_hour", "speed_kmh", "travel_time", "usual_delay_sec",
                   "delay_sec", "detector_speed", "detector_occupancy", "vehicle_count"]

        """
            Replace zero entries from FCD with the previous values for "volume per hour, speed and travel time".
        """
        fcd_df.volume_per_hour = fcd_df.loc[:, "volume_per_hour"].replace(to_replace=0, method='ffill')
        fcd_df.speed_kmh = fcd_df.loc[:, "speed_kmh"].replace(to_replace=0, method='ffill')
        fcd_df.travel_time = fcd_df.loc[:, "travel_time"].replace(to_replace=0, method='ffill')
        """ 
            Smooth noisy "volume per hour" measures from FCD due to max possible vehicle count per hour
            (2 sec for each vehicle).
        """
        fcd_df.loc[fcd_df.volume_per_hour > 1800, "volume_per_hour"] = 1800

        detectors_df = pd.read_fwf(detector_data_path, skiprows=[1], nrows=data_size)
        detectors_df.Vehicles = detectors_df.loc[:, "Vehicles"].replace(to_replace=0, method='ffill')
        """ 
            Smooth noisy measures from detector data due to max possible vehicle count per 5 min 
            (2 sec for each vehicle). 
        """
        detectors_df.loc[detectors_df["Vehicles"] > 150, ["Vehicles"]] = 150

        self.target = detectors_df["Vehicles"].to_numpy()
        self.detector_speed = detectors_df["Speed"].to_numpy()
        self.detector_occupancy = detectors_df["Occupancy"].to_numpy()

        """ 
            Transform vehicle count per five minutes to volume per hour.
        """
        self.target = self.target * 12

        aggregated_fcd = self.aggregate_fcd(fcd_df, start_timestamp, aggregate_time_range, data_size)
        missing_indexes = aggregated_fcd[12]
        """
            Remove missing feature indexes from detector data!!! 
            It's necessary to prevent any mismatching with the missing values at features.
        """
        self.target = np.delete(self.target, missing_indexes)
        self.detector_speed = np.delete(self.detector_speed, missing_indexes)
        self.detector_occupancy = np.delete(self.detector_occupancy, missing_indexes)

        self.averaged_data_df = pd.DataFrame(
            list(zip(aggregated_fcd[0], aggregated_fcd[1], aggregated_fcd[2], aggregated_fcd[3],
                     aggregated_fcd[4], aggregated_fcd[5], aggregated_fcd[6], self.detector_speed,
                     self.detector_occupancy, self.target)), columns=columns)

        self.split_data_df = pd.DataFrame(list(
            zip(aggregated_fcd[0], aggregated_fcd[1], aggregated_fcd[7], aggregated_fcd[8],
                aggregated_fcd[9], aggregated_fcd[10], aggregated_fcd[11], self.detector_speed,
                self.detector_occupancy, self.target)), columns=columns)

    def save_averaged_data_to_csv(self, path):
        self.averaged_data_df.to_csv(path, index=False, header=True)

    def save_split_data_to_csv(self, path):
        self.split_data_df.to_csv(path, index=False, header=True)

    """    
    
        Gets features from fcd per particular time range (per 5 minutes to match with detectors in this case)
        
        The average values of the entries within given time range for each feature :
        "volume_mean, speed_mean, travel_time_mean, usual_delay_mean, delay_mean"
        
        The closest entry values to the end timestamp:
        "volume_list, speed_list, travel_time_list, usual_delay_list, delay_list"
        
        If there is no entry for the given time range this index will be added to missing indexes and returned 
        to prevent any mismatching with the detector data.
    
    
    """

    def aggregate_fcd(self, df, begin_time_stamp, aggr_time, steps):
        speed_mean, volume_mean, travel_time_mean, usual_delay_mean, delay_mean = [], [], [], [], []
        speed_list, volume_list, travel_time_list, usual_delay_list, delay_list = [], [], [], [], []
        missing_indexes, date_list, timestamp_list = [], [], []

        for i in range(0, steps):
            results = df.loc[(df.time_stamp >= begin_time_stamp)
                             & (df.time_stamp <= begin_time_stamp + aggr_time)]
            if results.shape[0] > 0:
                timestamp_list.append(begin_time_stamp + aggr_time)
                date_list.append(results.iloc[- 1].date_time)
                """ average values within time range """
                volume_mean.append(results.volume_per_hour.mean())
                speed_mean.append(results.speed_kmh.mean())
                travel_time_mean.append(results.travel_time.mean())
                usual_delay_mean.append(results.usual_delay_sec.mean())
                delay_mean.append(results.delay_sec.mean())
                """ the closest item's values to the selected time"""
                volume_list.append(results.iloc[- 1].volume_per_hour)
                speed_list.append(results.iloc[- 1].speed_kmh)
                travel_time_list.append(results.iloc[- 1].travel_time)
                usual_delay_list.append(results.iloc[- 1].usual_delay_sec)
                delay_list.append(results.iloc[- 1].delay_sec)
            else:
                missing_indexes.append(i)

            begin_time_stamp = begin_time_stamp + aggr_time

        return date_list, timestamp_list, volume_mean, speed_mean, travel_time_mean, usual_delay_mean, delay_mean, \
               volume_list, speed_list, travel_time_list, usual_delay_list, delay_list, missing_indexes
