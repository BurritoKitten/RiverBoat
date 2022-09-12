"""
Is a class for holding the simulation environment. Specific simualations can be built on top of the base simulation

"""

# native packages
from abc import ABC, abstractmethod
from collections import OrderedDict
import os

# 3rd party packages
import yaml

# own packages
import src.Movers as Movers


class Environment(ABC):

    def __init__(self, h_params):
        self.h_params = h_params
        self.mover_dict = OrderedDict

    #@abstractmethod
    def initialize_environment(self):
        pass

    #@abstractmethod
    def reset_environment(self):
        pass

    #@abstractmethod
    def reset_baseline_environment(self):
        pass

    def add_mover(self, mover):
        """
        adds a mover to all of the movers in the simulation. The collection of movers are what are updated through time

        :param mover:
        :return:
        """
        if isinstance(mover,Movers.Mover):
            self.mover_dict[mover.name] = mover
        else:
            raise ValueError('Added mover must be of type mover')

    def run_simulation(self, is_baseline):
        """

        :param is_baseline: a boolean for if the episode being run is a baseline episode or a training episode is being
            run
        :return:
        """

        # reset environment

        # reset history of the movers


        step_num = 0
        t = 0
        delta_t = self.h_params['delta_t']
        max_t = self.h_params['max_t']

        while t < max_t:

            # step the simulation
            self.step()

            # add data to memory

            # add history of the movers the simulation

            # add simulation specific data from the learner

            t += delta_t
            step_num += 1

        # trim the history

    def step(self):
        pass
        # updates sensors

        # step movers

    def launch_training(self):
        """
        starts training. It either picks up where a previous training was stopped or ended, or
        it starts training from scratch. A dictionary in the environment class called 'h_params'
        holds all of the training parameters and settings. That can be loaded in the
        'create_environment' method. Data is regularly saved out from the training
        and simulations so the experimenter can analyze the results and check on the
        progress.

        :return:
        """

        if self.h_params['scenario']['continue_learning']:
            # load info and prepare simulation from old training to continue training
            pass

            # TODO implement

        else:
            # looking to start a training from scratch

            # check if data already exists for this scenario
            if os.path.exists("Output/"+str(self.h_params['scenario']['experiment_set'])):
                # check if trial number is there
                if os.path.exists("Output/"+str(self.h_params['scenario']['experiment_set'])+"/"+str(self.h_params['scenario']['trial_num'])):
                    overwrite_data = input("Data already exists for the experiment set and trial number. Do you want to overwrite the data (y/n)?")

                    if overwrite_data != 'y':
                        return

            self.create_folders()

        # loop over training
        elapsed_episodes = 0
        num_episodes = self.h_params['scenario']['num_episodes']
        while elapsed_episodes < num_episodes:



            elapsed_episodes += 1

    def create_folders(self):
        """
        creates all of the sub folders that are needed for the training to save data to.
        Training progress, simulation telemetry, and models are all saved for later use
        and analysis.

        :return:  void
        """

        # create base folder
        try:
            os.mkdir("Output/" + str(self.h_params['scenario']['experiment_set']))
        except:
            pass

        # create trial folder
        try:
            os.mkdir("Output/"+str(self.h_params['scenario']['experiment_set'])+"/"+str(self.h_params['scenario']['trial_num']))
        except:
            pass

        # create sub folders
        sub_folder_names = ["\Models","\TrainingHistory","\TrainingHistory\Data","\TrainingHistory\Graphs",
                            "\Baseline","\Baseline\Data","\Baseline\Graphs",
                            "\RobustBaseline", "\RobustBaseline\Data", "\RobustBaseline\Graphs",
                            "\Progress","\Progress\Data","Progress\Graphs"]
        for sfn in sub_folder_names:
            try:
                os.mkdir("Output/"+str(self.h_params['scenario']['experiment_set'])+"/"+str(self.h_params['scenario']['trial_num'])+sfn)
            except:
                pass

    @staticmethod
    def create_environment(file_name):
        """
        Given a yaml file that has configuration details for the environment and hyperparameters for the agents,
        a dictionary is built and returned to the user

        :param file_name:
        :return:
        """
        with open(file_name, "r") as stream:
            try:
                hp_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # create environment based on the hyper_parameters
        env = Environment(hp_data)

        return env