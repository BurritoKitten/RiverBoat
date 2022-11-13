
# native packages
from abc import ABC, abstractmethod
from collections import namedtuple

# 3rd party packages
import numpy as np

# own packages


class Controller(ABC):

    def __init__(self, coeffs: namedtuple):
        """
        a super class for controllers that work to keep the agent following a path.

        :param coeffs: a named tuple that has the fixed controller gain coefficients.
        """
        self.coeffs = coeffs

    @abstractmethod
    def calculate_error(self):
        """
        useing the state information and a desired point to get to, find the error relative to that
        :return: the error
        """
        pass

    @abstractmethod
    def get_steering_change(self, dt, state):
        """
        given the time step and current state information, return the propeller angle change for the boat. The
        controller uses the state and the path to control what to actuate the actuator too.

        :param dt: time step size [s]
        :param state: a dictionary containing the current state information
        :return: propeller angle change [rad]
        """
        pass

class PDController(Controller):

    def __init__(self, coeffs: namedtuple, is_clipped: bool, max_change: float):
        super().__init__(coeffs)
        self.is_clipped = is_clipped
        self.max_change = max_change
        self.old_error = 0.0
        self.error = 0.0

    def calculate_error(self, state):

        x = state['x']
        y = state['y']
        phi = state['phi']

        rot_mat = [[np.cos(phi), np.sin(phi), 0.0],
                   [-np.sin(phi), np.cos(phi), 0.0],
                   [0.0, 0.0, 1.0]]
        rot_mat = np.reshape(rot_mat, (3, 3))

        #diff = np.subtract(nav_point[0:3], [x,y,phi])
        diff = 0

        error = np.matmul(rot_mat, diff)

        self.error = error[1] # error in local y dimension

    def get_steering_change(self, dt, state):

        self.calculate_error(state)

        y_dot = (self.error - self.old_error) / dt

        prop = self.coeffs['p'] * self.error[1] / state['v_mag']
        deriv = self.coeffs['d']* y_dot

        steering_change = prop + deriv

        if self.is_clipped:
            if np.abs(steering_change) > np.deg2rad(self.max_change * dt):
                steering_change = np.deg2rad(self.max_change * dt) * np.sign(steering_change)

        self.old_error = self.error

        return -steering_change
