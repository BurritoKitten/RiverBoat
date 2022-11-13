"""
Is a class for holding the simulation environment. Specific simualations can be built on top of the base simulation

"""

# native packages
from abc import ABC, abstractmethod
from collections import namedtuple
from collections import OrderedDict
import os

# 3rd party packages
import numpy as np
import pandas as pd
import yaml

# own packages
import src.ActionOperation as ActionOperation
import src.Controller as Controller
import src.LearningAlgorithms as LearningAlgorithms
import src.Movers as Movers
import src.Sensors as Sensors


class Environment(ABC):

    def __init__(self, h_params):
        self.h_params = h_params
        self.mover_dict = OrderedDict()
        self.agent = None # learning agent
        self.ao = None # action operation. Used to convert raw outputs to inputs

    #@abstractmethod
    def initialize_environment(self):
        pass

    #@abstractmethod
    def reset_environment(self, reset_to_max_power):
        """
        generates a random initial state for the simulaiton.
        :return:
        """

        domain = self.h_params['scenario']['domain']

        # set a random initial location for the destination
        x = np.random.random() * domain
        y = np.random.random() * domain
        self.destination = [x,y]

        # set up random locations and orientations for the movers
        for name, mover in self.mover_dict.items():

            if 'river_boat' in name:

                # whipe state to base configuration for the boat
                #mover.initalize_in_state_dict()

                # reset the river boats data
                dst_to_dest = 0.0
                while dst_to_dest <= 10.0:
                    mover.state_dict['x_pos'] = np.random.random() * domain
                    mover.state_dict['y_pos'] = np.random.random() * domain
                    dst_to_dest = np.sqrt((mover.state_dict['x_pos'] - self.destination[0]) ** 2 + (
                                mover.state_dict['y_pos'] - self.destination[1]) ** 2)

                mover.state_dict['psi'] = np.random.random() * 2.0*np.pi
                mover.state_dict['delta'] = (np.random.random()-0.5)*2.0*np.abs(mover.state_dict['delta_max'][0])

                # reset the velocities of the boat
                mover.state_dict['v_xp'] = np.random.random()
                mover.state_dict['v_yp'] = np.random.random()-0.5
                mover.state_dict['psi_dot'] = np.random.random()-0.5

                # power setting needs to be set from action designation.
                if reset_to_max_power:
                    mover.state_dict['power'] = mover.state_dict['power_max']
                else:
                    mover.state_dict['power'] = np.random.random()*mover.state_dict['power_max']

                # reset the fuel on the boat
                mover.state_dict['fuel'] = mover.state_dict['fuel_capacity']

            elif 'static_circle' in name:
                # reset the static circle
                max_dst_to_other = 0.0
                dst_to_dest = 0.0
                while dst_to_dest*2.0 <= mover.state_dict['radius'] and max_dst_to_other*2.0 <= mover.state_dict['radius']:
                    # draw random positions for the obstacle until it is not too close to the goal location or any other
                    # movers

                    mover.state_dict['x_pos'] = np.random.random()*domain
                    mover.state_dict['y_pos'] = np.random.random() * domain
                    dst_to_dest = np.sqrt( (mover.state_dict['x_pos']-self.destination[0])**2 + (mover.state_dict['y_pos']-self.destination[1])**2)

                    max_dst_to_other = np.infty

                    for tmp_name, tmp_mover in self.mover_dict.items():
                        if tmp_name != name:
                            # other obstacle selected in the loop
                            x_other = tmp_mover.state_dict['x_pos']
                            y_other = tmp_mover.state_dict['y_pos']
                            tmp_dst = np.sqrt((mover.state_dict['x_pos']-x_other)**2 + (mover.state_dict['y_pos']-y_other)**2)
                            if tmp_dst < max_dst_to_other:
                                max_dst_to_other = tmp_dst
            else:
                raise ValueError('Mover not currently supported')

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
            name = mover.state_dict['name']
            self.mover_dict[name] = mover
        else:
            raise ValueError('Added mover must be of type mover')

    def run_simulation(self, is_baseline, reset_to_max_power):
        """

        :param is_baseline: a boolean for if the episode being run is a baseline episode or a training episode is being
            run
        :return:
        """

        # reset environment
        if is_baseline:
            self.reset_baseline_environment(reset_to_max_power)
        else:
            self.reset_environment(reset_to_max_power)

        # reset the time stamps of the
        step_num = 0
        t = 0
        delta_t = self.h_params['scenario']['time_step']
        max_t = self.h_params['scenario']['max_time']

        # reset history of the movers
        for name, mover in self.mover_dict.items():
            mover.reset_history(int(np.ceil(max_t/delta_t)))

        while t < max_t:

            # step the simulation
            self.step(t)

            # add data to memory

            # add history of the movers the simulation

            # add simulation specific data from the learner

            t += delta_t
            step_num += 1

        # trim the history
        self.history.drop(range(step_num, len(self.history)), inplace=True)
        for name, mover in self.mover_dict.items():
            mover.trim_history(step_num)

    def step(self,t):
        """
        steps the simulation one time step. The agent is given a normalized state to make an action selection. Then
        the action is operated on the environment

        :param t:  time of the simulation [s]
        :return:
        """

        # updates sensors
        for name, mover in self.mover_dict.items():
            mover.update_sensors(self.mover_dict)
            mover.derived_measurements(self.destination)

        # step movers
        for name, mover in self.mover_dict.items():
            if mover.can_learn:
                # this mover is an agent and not a deterministic entity

                # normalize the state
                keys = mover.observation_df['name']
                dimensional_values = OrderedDict()
                non_dimensional_values = OrderedDict()
                for key in keys:
                    dimensional_values[key] = mover.state_dict.get(key)
                    non_dimensional_values[key] = mover.state_dict.get(key)

                norm_values = mover.observation_df['norm_value'].to_numpy()
                norm_methods = mover.observation_df['norm_method'].to_numpy()
                for i, norm_strat in enumerate(norm_values):
                    # TODO norm by strategy. MOve this to its own method
                    non_dimensional_values[keys[i]] = dimensional_values[keys[i]]/norm_values[i]

                # add the normalized sensor measurements
                for sensor in mover.sensors:
                    norm_meas = sensor.get_norm_measurements()
                    non_dimensional_values.update(norm_meas)

                # get the action and network outputs from the network
                inp = list(non_dimensional_values.values())
                raw_ouputs = self.agent.get_output(inp)

                # convert the action to a command change
                propeller_angle_change, power_change = self.ao.action_to_command(t,mover.state_dict, raw_ouputs)

                # apply the actuator changes to the mover
                power = power_change + mover.state_dict['power']
                propeller_angle = propeller_angle_change + mover.state_dict['delta']
                mover.set_control(power,propeller_angle)

            mover.step(time=t)

        # normalize the state prime

        # get the reward

        # set if the simulation has reach a termination condition

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
                    overwrite_data = input("Data already exists for the experiment set and trial number. Do you want to overwrite the data [y/n]? ")

                    if overwrite_data != 'y':
                        return

            self.create_folders()

        # save a copy of the hyperparameters
        self.save_hparams()

        # add agents/entities to the simulation. Boats, and obstacles
        self.add_entities()

        # load in the senors
        sensors = self.add_sensors()

        # get the controller
        controller = self.get_controller()

        # get the action determination
        action_size, ao, reset_to_max_power = self.get_action_size_and_operation(controller)
        self.ao = ao

        # get the state size
        state_size = self.get_state_size()

        # temporary for when only using one agent
        #if len(state_size) != 1:
        #    raise ValueError('Only single agent learning currently supported')

        # get the learning algorithm
        la = self.get_learning_algorithm(action_size,state_size)
        self.agent = la

        # loop over training
        elapsed_episodes = 0
        num_episodes = self.h_params['scenario']['num_episodes']
        while elapsed_episodes < num_episodes:

            # run baseline episodes if required

            # run the episode where training data is accumulated
            is_baseline = False
            self.run_simulation(is_baseline, reset_to_max_power)

            # write episode history out to a file

            # train the networks

            # update target networks if applicable

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
                            "\Progress","\Progress\Data","\Progress\Graphs"]
        for sfn in sub_folder_names:
            try:
                os.mkdir("Output/"+str(self.h_params['scenario']['experiment_set'])+"/"+str(self.h_params['scenario']['trial_num'])+sfn)
            except:
                pass

        # create initial info file for the user to type to help document the trial
        new_file = True
        if os.path.exists("Output/"+str(self.h_params['scenario']['experiment_set'])+"/"+str(self.h_params['scenario']['trial_num'])+'notes.txt'):
            overwrite_notes = input("Do you want to overwrite the notes file [y/n]?")

            if overwrite_notes != 'y':
                new_file = False

        if new_file:
            with open("Output/"+str(self.h_params['scenario']['experiment_set'])+"/"+str(self.h_params['scenario']['trial_num'])+'/notes.txt', 'w') as f:

                f.write("TODO, the user should add some comments about the reason for this scenario and about some of the "
                        "findings from this trial. This will help for compiling results.")

    def add_sensors(self):
        """
        creates sensor objects for all of the sensors defined in the input file. The sensors can then be installed into
        movers/agents

        :return: none
        """

        sensors_data = self.h_params['sensors']

        for name, sensor in sensors_data.items():

            if 'lidar_' in name:
                # create a ProcessedLidar object that simulates the measurements after the raw lidar data has been
                # filtered and obstacles more succintly defined

                measurement_df = pd.DataFrame(columns=['name','norm_value','norm_method'])
                meas_var_info = sensor['max_range']
                meas_var_info = meas_var_info.split(',')
                row = dict()
                row['name'] = 'max_range'
                row['norm_value'] = float(meas_var_info[1])
                row['norm_method'] = meas_var_info[2]
                measurement_df = measurement_df.append(row, ignore_index=True)

                meas_var_info = sensor['max_theta']
                meas_var_info = meas_var_info.split(',')
                row = dict()
                row['name'] = 'max_theta'
                row['norm_value'] = float(meas_var_info[1])
                row['norm_method'] = meas_var_info[2]
                measurement_df = measurement_df.append(row, ignore_index=True)

                tmp_sensor = Sensors.ProcessedLidar(base_range=float(sensor['base_range']),base_theta=float(sensor['base_theta']),
                                       name=name,measurement_norm_df=measurement_df,mover_owner_name=sensor['install_on'])

                for mover_name, mover in self.mover_dict.items():
                    if mover_name == tmp_sensor.mover_owner_name:
                        mover.add_sensor(tmp_sensor)

            else:
                raise ValueError('Sensor not currently supported')

    def add_entities(self):
        """
        Adds movers and obstacles to the simulation to run

        :return:
        """
        movers = self.h_params['movers']
        for name, mover_details in movers.items():

            if 'river_boat' in name:
                # create a river boat mover and add it to the mover dictionary
                tmp_mover = Movers.RiverBoat.create_from_yaml(name,mover_details)
            elif 'static_circle' in name:
                # create a circle obstacle that does not move
                tmp_mover = Movers.StaticCircleObstacle.create_from_yaml(name,mover_details)
            else:
                raise ValueError('Mover type ' + name + ' not currently supported')

            self.add_mover(tmp_mover)

    def get_controller(self):
        """
        the controller coeffiecints from the input file are loaded so the boat can use it.
        Currently only, a pd controller is supported.

        :return: the controller object
        """
        controller_info = self.h_params['controller']
        if controller_info['type'] == 'pd':
            coeffs_lst = controller_info['coeffs'].split(',')
            coeffs = namedtuple("Coeffs", "p d")
            pd_coeffs = coeffs(coeffs_lst[0], coeffs_lst[1])
            controller = Controller.PDController(coeffs=pd_coeffs, is_clipped=False, max_change=np.pi)
        else:
            raise ValueError('Controller type not currently supported')

        return controller

    def get_action_size_and_operation(self, controller):
        """
        from the input file, get the action formulation specified by the user. The action specification dictates how
        an action is split. For example, are control points used to build a path, are a set of canned paths used, or
        is direct control of the actuators used. The output then is the action formulation object and the number
        of outputs a neural network will need. The action operation is also used during the simuliation to convert
        raw neural network outputs to actuator control changes.

        :param controller: the controller used by the boat. This can be None if the neural network is directly
            controlling the actuators.
        :return: the number of actions a neural network needs to predict, and the action operation object that converts
            neural network outputs to actuator control changes.
        """
        action_type = self.h_params['action_description']['action_type']
        action_designation = self.h_params['action_description']['action_designation']
        ao = None

        if action_type == 'continous' and action_designation == 'control_point':
            # use continously defined control points to define a b-spline path
            ao = ActionOperation.PathContinousCp(name='discrete_cp',
                                                replan_rate=self.h_params['action_description']['replan_rate'],
                                                controller=controller,
                                                angle_range=self.h_params['action_description']['angle_range'],
                                                power_range=self.h_params['action_description']['power_range'])
        elif action_type == 'continous' and action_designation == 'direct':
            # agent is directly controlling the actuators with discrete choices
            ao = ActionOperation.DirectControlActionContinous(name='discrete_direct',
                                                             max_propeller=self.h_params['action_description'][
                                                                 'max_propeller'],
                                                             max_power=self.h_params['action_description'][
                                                                 'max_power'])
        elif action_type == 'discrete' and action_designation == 'canned':
            # used discrete canned paths
            ao = ActionOperation.PathDiscreteCanned(name='canned_cp',replan_rate=self.h_params['action_description']['replan_rate'],
                                                controller=controller,
                                                turn_amount_lst=self.h_params['action_description']['turn_values'],
                                                s_curve_lst=self.h_params['action_description']['s_curve_values'],
                                                power_change_lst=self.h_params['action_description']['power_values'])
        elif action_type == 'discrete' and action_designation == 'control_point':
            # use disrete CP action type
            ao = ActionOperation.PathDiscreteCp(name='discrete_cp',
                                                replan_rate=self.h_params['action_description']['replan_rate'],
                                                controller=controller,
                                                angle_adj_lst=self.h_params['action_description']['angle_values'],
                                                power_change_lst=self.h_params['action_description']['power_values'])

        elif action_type == 'discrete' and action_designation == 'direct':
            # agent is directly controlling the actuators with discrete choices
            ao = ActionOperation.DirectControlActionDiscrete(name='discrete_direct',
                                                prop_change_angle_lst = self.h_params['action_description']['propeller_values'],
                                                power_change_lst = self.h_params['action_description']['power_values'])

        else:
            raise ValueError('Action type and designation combo not supported')

        # get the size of the
        action_size = ao.get_action_size()
        reset_to_max_power = ao.reset_to_max_power()
        return action_size, ao, reset_to_max_power

    def get_state_size(self):
        """
        for a specified mover, the observation space size/enumeration is calculated

        :return:
        """

        n_obs = 0

        #movers = self.h_params['movers']
        for name, mover in self.mover_dict.items():

            # only learning movers observe state information currently
            if mover.can_learn:

                n_observations = len(mover.observation_df)

                # also get the measurements generated by the sensors
                for sensor in mover.sensors:
                    #sensor.calc_measurements(self.mover_dict)
                    n_observations += sensor.n_measurements

                    #mover.observation_df = pd.concat([mover.observation_df, sensor.measurement_norm_df], ignore_index=True)

                n_obs = n_observations

        return n_obs

    def get_learning_algorithm(self, action_size, state_size):
        """
        Gets the learning algorithm that will drive how the agent is trained. Example: DQN, DDPG, etc.

        :param action_size: integer for the number of actions the agent will have to output from its neural network
        :param state_size: interger for the observation space that the agent will accept into its neural network
        :return: the built learning algorithm object. Serves as the agent as well.
        """

        settings = self.h_params['learning_algorithm']

        activation = settings['activation']
        last_activation = settings['last_activation']
        layer_numbers = settings['layers']
        layer_numbers = layer_numbers.split(',')
        layer_numbers = [float(i) for i in layer_numbers]
        loss = settings['loss']

        if settings['name'] == 'DQN':
            la = LearningAlgorithms.DQN(action_size, activation, self.h_params, last_activation, layer_numbers, loss, state_size)
        elif settings['name'] == 'DDPG':
            la = None
        else:
            raise ValueError('Learning algorithm currently not supported')

        return la

    def save_hparams(self):
        """
        saves a copy of the hyperparameters used in the simulation for documentation and reference later. The parameters
        are saved in an unchanged state to the trial directory.
        :return:
        """
        with open("Output/"+str(self.h_params['scenario']['experiment_set'])+"/"+str(self.h_params['scenario']['trial_num'])+'/hyper_parameters.yaml', 'w') as file:
            yaml.safe_dump(self.h_params,file)

    @staticmethod
    def create_environment(file_name):
        """
        Given a yaml file that has configuration details for the environment and hyperparameters for the agents,
        a dictionary is built and returned to the user

        :param file_name: a file name that has all of the hyperparameters. This should be a yaml file
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