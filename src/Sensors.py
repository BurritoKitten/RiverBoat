"""
Is a class for defining sensors that each of the movers/robots can have

"""

# native packages
from abc import ABC, abstractmethod
from collections import OrderedDict

# 3rd party packages
import numpy as np
import pandas as pd

# own packages


class Sensor(ABC):

    def __init__(self, name, mover_owner_name):
        """
        base class for all sensors used in the simulation
        :param name: string name of the sensor
        :param mover_owner_name: string name of the mover that the sensor is on/installed on
        """
        self.name = name
        self.mover_owner_name = mover_owner_name
        self.measurement_dict = OrderedDict()

    @abstractmethod
    def init_measurement_dict(self):
        """
        Each child must invoke this method to prepare the measurement dictionary to hold both the
        :return:
        """
        pass

    @abstractmethod
    def calc_measurements(self, mover_dict):
        """
        abstract class that takes the current state of the simulation and interally updates its own measurements

        :param mover_dict: ordered dictionary that has all of the movers in the simulation
        :return:
        """
        pass

    @abstractmethod
    def get_raw_measurements(self):
        """
        gets the current raw measurements the sensor has taken

        :return:
        """
        pass

    @abstractmethod
    def get_norm_measurements(self):
        """
        gets the current normalized measuresments the sensor has taken
        :return:
        """
        pass


class ProcessedLidar(Sensor):

    def __init__(self,base_range, base_theta,name, measurement_norm_df, mover_owner_name):
        """
        a sensor that observes the distance and angle of an object. It is assumed that beams would be used to get data
        and the outout of the sensors is what some other processing is done to get the returned features.

        :param base_range: the range returned if the sensor does not see an object
        :param base_theta: the angle to an obstacle that is returned if it does not see a mover
        :param name: a string for the name of the lidar sensor
        :param max_range: the maximum range of the sensor
        :param mover_owner_name:
        """
        super().__init__(name, mover_owner_name)
        self.measurement_norm_df = measurement_norm_df
        self.base_range = base_range
        self.base_theta = base_theta

        self.init_measurement_dict()

    def init_measurement_dict(self):
        """
        initializes the measurement dictionary that has the current raw and normalized measurements. The lidar sensor
        has both a direction and distance to the other objects it observes.

        :return:
        """

        self.measurement_dict['theta'] = 0.0  # [rad]
        self.measurement_dict['theta_norm'] = 0.0  # [rad]
        self.measurement_dict['dist'] = 0.0  # [m]
        self.measurement_dict['dist_norm'] = 0.0  # [m]
        self.n_measurements = 2

    def calc_measurements(self, mover_dict):
        """
        gets the angle and distance to all other movers in the simulation. All the measurements are relative to the
        mover the sensor belongs too.

        :param mover_dict: ordered dictionary that has all of the movers in the simulation
        :return: an ordered dictionary for the measurements the lidar taks
        """
        own_mover = mover_dict[self.mover_owner_name]
        x_own = own_mover.state_dict['x_pos']
        y_own = own_mover.state_dict['y_pos']

        # dictionary of measured observations

        for name, mover in mover_dict.items():
            if name != self.mover_owner_name:
                # lidar does not obesrve it self. Get the state information it wants

                x_other = mover.state_dict['x_pos']
                y_other = mover.state_dict['y_pos']

                theta = np.arctan2(y_other-y_own,x_other-x_own)

                dst = np.sqrt( (x_other-x_own)**2 + (y_other-y_own)**2 )

                max_range_row = self.measurement_norm_df[self.measurement_norm_df['name'] == 'max_range']
                if dst > max_range_row['norm_value'].iloc[0]:
                    self.measurement_dict['theta'] = self.base_theta
                    self.measurement_dict['dist'] = self.base_range
                else:

                    self.measurement_dict['theta'] = theta
                    self.measurement_dict['dist'] = dst

        # normalize and save the normalized measurements
        #tmp = self.measurement_norm_df[self.measurement_norm_df['name']=='max_theta']['norm_value'].to_numpy()[0]
        self.measurement_dict['theta_norm'] = self.measurement_dict['theta']/(self.measurement_norm_df[self.measurement_norm_df['name']=='max_theta']['norm_value'].to_numpy()[0])
        self.measurement_dict['dist_norm'] = self.measurement_dict['dist']/(self.measurement_norm_df[self.measurement_norm_df['name']=='max_range']['norm_value'].to_numpy()[0])

    def get_raw_measurements(self):
        """
        gets the current measurements in their native units
        :return:
        """
        raw = dict((k, self.measurement_dict[k]) for k in ('theta', 'dist') if k in self.measurement_dict)
        raw[self.name + '_theta'] = raw.pop('theta')
        raw[self.name + '_dist'] = raw.pop('dist')
        return raw

    def get_norm_measurements(self):
        """
        gets the current measurements in their native units
        :return:
        """
        raw = dict((k, self.measurement_dict[k]) for k in ('theta_norm', 'dist_norm') if k in self.measurement_dict)
        raw[self.name + '_theta'] = raw.pop('theta_norm')
        raw[self.name + '_dst'] = raw.pop('dist_norm')
        return raw