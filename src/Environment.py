"""
Is a class for holding the simulation environment. Specific simualations can be built on top of the base simulation

"""

from abc import ABC, abstractmethod


class Environment(ABC):

    def __init__(self, h_params):
        self.h_params = h_params

    @abstractmethod
    def initialize_environment(self):
        pass

    @abstractmethod
    def reset_environment(self):
        pass

    @abstractmethod
    def reset_baseline_environment(self):
        pass

    def run_simulation(self):

        t = 0
        delta_t = self.h_params['delta_t']
        max_t = self.h_params['max_t']

        while t < max_t:

            # updates sensors

            # step movers

            t += delta_t


