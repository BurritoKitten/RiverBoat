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
import torch
import yaml

# own packages
import src.ActionOperation as ActionOperation
import src.Controller as Controller
import src.LearningAlgorithms as LearningAlgorithms
import src.Movers as Movers
import src.ReplayMemory as ReplayMemory
import src.RewardFunctions as RewardFunctions
import src.Sensors as Sensors


class Environment(ABC):

    def __init__(self, h_params):
        self.h_params = h_params
        self.mover_dict = OrderedDict()
        self.agent = None  # learning agent
        self.ao = None  # action operation. Used to convert raw outputs to inputs
        self.reward_func = None  # reward function for the simulation
        self.device = 'cuda' # TODO need to check this
        self.header = ['time', 'reward', 'is_terminal', 'is_crashed', 'is_reached'] # history data frame header
        self.history = None  # eventually a data frame to hold information about the simulation

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
                mover.initalize_in_state_dict()

                # reset the river boats data
                dst_to_dest = 0.0

                while dst_to_dest <= 10.0:
                    mover.state_dict['x_pos'] = np.random.random() * domain
                    mover.state_dict['y_pos'] = np.random.random() * domain
                    dst_to_dest = np.sqrt((mover.state_dict['x_pos'] - self.destination[0]) ** 2 + (
                                mover.state_dict['y_pos'] - self.destination[1]) ** 2)

                mover.state_dict['psi'] = np.random.random() * 2.0*np.pi
                tmp_delta = (np.random.random()-0.5)*2.0*np.abs(mover.state_dict['delta_max'][0])

                # reset the velocities of the boat
                mover.state_dict['v_xp'] = np.random.random()
                mover.state_dict['v_yp'] = np.random.random()-0.5
                mover.state_dict['psi_dot'] = 0.0 #(np.random.random()-0.5)/10.0

                # power setting needs to be set from action designation.
                if reset_to_max_power:
                    tmp_power = mover.state_dict['power_max']
                else:
                    tmp_power = np.random.random()*mover.state_dict['power_max']

                mover.set_control(tmp_power, tmp_delta)

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

        # updates sensors
        for name, mover in self.mover_dict.items():
            mover.update_sensors(self.mover_dict)
            mover.derived_measurements(self.destination)

        # reset reward function information
        self.reward_func.reset(self.mover_dict)

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
        :param ep_num: the episode number in the training
        :return:
        """

        # reset environment
        if is_baseline:
            self.reset_baseline_environment(reset_to_max_power)
        else:
            self.reset_environment(reset_to_max_power)

        # reset the reward function
        self.reward_func.reset(self.mover_dict)

        # reset the time stamps of the
        step_num = 0
        t = 0.0
        delta_t = self.h_params['scenario']['time_step']
        max_t = float(self.h_params['scenario']['max_time'])
        max_steps = int(np.ceil(max_t / delta_t))

        # reset history of the movers
        for name, mover in self.mover_dict.items():
            mover.reset_history(max_steps)

        # reset own history
        self.history = pd.DataFrame(data=np.zeros(((max_steps), len(self.header))),columns=self.header)

        # holding the state prior to a step. Initialized as none so the loop knows to use the first state as its state
        state = None
        action = None
        reward = 0.0
        cumulative_reward = 0.0
        # boolean for if the agent has reached a terminal state prior to the end of the episode
        is_terminal = False
        end_step = True  # if this is true the agent will know to make another prediction
        while step_num < max_steps and not is_terminal:

            # step the simulation
            interim_state, interim_action, interim_reward, interim_next_state, end_step, is_terminal, is_crashed, is_success = self.step(t, end_step)

            # if the state is empty update it to the interim state
            if state is None:
                state = interim_state
                action = interim_action
                reward = 0.0

            if end_step or is_terminal:
                # add data to memory because the agents step has completed.
                next_state = interim_next_state

                # convert the tuple to tensors
                state_tensor = ReplayMemory.convert_numpy_to_tensor(self.device,list(state.values()))
                next_state_tensor = ReplayMemory.convert_numpy_to_tensor(self.device, list(next_state.values()))
                action_tensor = torch.tensor(action)
                reward_tensor = torch.tensor(reward)
                is_terminal_tensor = torch.tensor(is_terminal)

                # store the data
                self.replay_storage.push(state_tensor,action_tensor,next_state_tensor,reward_tensor,is_terminal_tensor)

                # reset the state to None for the agents next step
                state = None
                action = None
                reward = 0.0
            else:
                reward += interim_reward

            # add history of the movers the simulation
            for name, mover in self.mover_dict.items():
                mover.add_step_history(step_num)

            # add simulation specific history
            action = interim_action
            telemetry = np.concatenate(([t, reward, is_terminal, is_crashed, is_success],[action]))
            self.history.iloc[step_num] = telemetry

            # add simulation specific data from the learner. destination distance

            t += delta_t
            step_num += 1
            cumulative_reward += interim_reward

        # trim the history
        self.history.drop(range(step_num, len(self.history)), inplace=True)
        for name, mover in self.mover_dict.items():
            mover.trim_history(step_num)

        # sort the stored data into buffers as needed
        if not is_baseline:
            self.replay_storage.sort_data_into_buffers()

    def write_history(self, ep_num):
        """
        writes the histories of the movers and the simulation out for later analysis
        :return:
        """

        total_history = self.history

        for name, mover in self.mover_dict.items():

            # get mover files
            tmp_histroy = mover.history

            # add history together
            total_history = pd.concat([total_history, tmp_histroy], axis=1)

        # write total history out to a file
        file_name = 'Output//' + str(self.h_params['scenario']['experiment_set'])+ '//' + str(self.h_params['scenario']['trial_num'])+'//TrainingHistory//Data//History_'+str(ep_num)+'.csv'
        total_history.to_csv(file_name, index=False)

    def step(self, t, end_step):
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
                propeller_angle_change, power_change, action, end_step = self.ao.action_to_command(t,mover.state_dict, raw_ouputs)

                # apply the actuator changes to the mover
                power = power_change + mover.state_dict['power']
                propeller_angle = propeller_angle_change + mover.state_dict['delta']
                mover.set_control(power,propeller_angle)

            mover.step(time=t)

        # save the original state before stepping
        state = non_dimensional_values

        # normalize the state prime
        for name, mover in self.mover_dict.items():
            mover.update_sensors(self.mover_dict)
            mover.derived_measurements(self.destination)

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

        next_state = non_dimensional_values

        # get the reward
        reward = self.reward_func.get_reward(t, self.mover_dict)

        # get if the simulation has reach a termination condition
        is_terminal = self.reward_func.get_terminal()

        return state, action, reward, next_state, end_step, is_terminal, self.reward_func.is_crashed, self.reward_func.is_success

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
        action_size, self.ao, reset_to_max_power = self.get_action_size_and_operation(controller)

        # get the state size
        state_size = self.get_state_size()

        # temporary for when only using one agent
        #if len(state_size) != 1:
        #    raise ValueError('Only single agent learning currently supported')

        # get the learning algorithm
        la = self.get_learning_algorithm(action_size,state_size)
        self.agent = la

        # get the reward function
        self.reward_func = RewardFunctions.select_reward_function(self.h_params, self.ao)

        # get the replay buffer
        self.get_memory_mechanism()

        # loop over training
        elapsed_episodes = 0
        num_episodes = self.h_params['scenario']['num_episodes']
        while elapsed_episodes < num_episodes:

            print("Episode Number {}".format(elapsed_episodes))

            # run baseline episodes if required

            # run the episode where training data is accumulated
            is_baseline = False
            self.run_simulation(is_baseline, reset_to_max_power)

            # write episode history out to a file
            self.write_history(elapsed_episodes)

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

                        # add the data from the sensor to the history for the mover
                        tmp_state = tmp_sensor.get_raw_measurements()
                        for ts in tmp_state:
                            mover.history_header.append(mover.state_dict['name'] + '_' + ts)

            else:
                raise ValueError('Sensor not currently supported')

    def add_entities(self):
        """
        Adds movers and obstacles to the simulation to run

        :return:
        """
        movers = self.h_params['movers']
        delta_t = self.h_params['scenario']['time_step']
        for name, mover_details in movers.items():

            if 'river_boat' in name:
                # create a river boat mover and add it to the mover dictionary
                tmp_mover = Movers.RiverBoat.create_from_yaml(name,mover_details, delta_t)
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
        selection_size = ao.get_selection_size()
        reset_to_max_power = ao.reset_to_max_power()

        # add columns to the history header for the simulation
        for i in range(selection_size):
            self.header.append('a'+str(i))

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

    def get_memory_mechanism(self):
        """
        parses the input file to create the replay buffer that is described in hte input file.
        The replay buffer stores data for the learning agent to use to learn the solution.

        :return:
        """

        memory_info = self.h_params['replay_data']

        self.replay_storage = ReplayMemory.ReplayStorage(capacity=memory_info['capacity'], extra_fields=[], strategy=memory_info['replay_strategy'])

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