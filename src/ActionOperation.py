

# native packages
from abc import ABC, abstractmethod

# 3rd party packages
import numpy as np
from scipy.special import comb


def bernstein_poly(i, n, t):
    """
    helper that generates bernstein polynomials
    """
    return comb(n, i) * ( np.power(t,(n-i)) ) * np.power(1.0 - t,i)


def bezier_curve(control_points, n_samples):
    """
    Given a list of (x, y) points that serve as control points, a path is generated. The path is created with
    n_samples of points to define the path.

    :param control_points: an array of x,y control points that define a bezier curve
    :param n_samples: the number of samples to collect spanning 0 to 1
    """

    n_cp = len(control_points)

    t = np.linspace(0.0, 1.0, n_samples)

    pa = np.zeros((n_cp,n_samples))
    for i in range( n_cp):
        pa[i,:] = (bernstein_poly(i, n_cp-1, t))
    pa = np.reshape(pa,(len(pa),len(pa[0])))

    samples = np.zeros((n_samples,4))
    samples[:,0] = np.dot(control_points[:,0], pa) # x
    samples[:,1] = np.dot(control_points[:,1], pa) # y

    # calculate angles
    for i in range(len(samples)):
        if i == len(samples) - 1:
            tmp_vec = [samples[i, 0] - samples[i - 1, 0], samples[i, 1] - samples[i - 1, 1]]
        elif i == 0:
            tmp_vec = [samples[i + 1, 0] - samples[i, 0], samples[i + 1, 1] - samples[i, 1]]
        else:
            tmp_vec = [samples[i + 1, 0] - samples[i - 1, 0], samples[i + 1, 1] - samples[i - 1, 1]]

        samples[i,2] = np.arctan2(tmp_vec[1],tmp_vec[0])

    samples = np.flip(samples, axis=0)

    return samples


class ActionOperation(ABC):

    def __init__(self, name):
        """
        action operation is a super class that provides a single interface for converting an agents raw output into
        steering and throttle setting changes. All the children implement the action space and formulation
        differently.

        :param name: a string for the action. This is helpful for logging and debugging
        """
        self.name = name

    @abstractmethod
    def action_to_command(self, time, state, input):
        """
        the child class needs to implement a function that takes in a state, and produces a command to for the aircraft.
        The command can also use internal data/items to generate the command with the state.
        :return:
        """
        pass

    @abstractmethod
    def get_action_size(self):
        """
        An agent will need to produce action choices. All agents are acting on the same boat. This tells the agent
        the number of outputs needed to use the action formulation
        :return: an integer for the number of outputs
        """
        pass


class DirectControlActionDiscrete(ActionOperation):

    def __init__(self, name, prop_change_angle_lst, power_change_lst=None):
        """
        a class that converts an integer class output from a neural network to a mapped propeller change
        and a powr change. The power does not need to change

        :param name:
        :param prop_change_angle_lst:
        :param power_change_lst:
        """
        super().__init__(name)
        if prop_change_angle_lst is None or len(prop_change_angle_lst) == 0:
            raise ValueError("The propeller change angle list cannot be none and must be have at least one element")
        prop_change_angle_lst = prop_change_angle_lst.split(',')
        self.prop_change_angle_lst = [float(i) for i in prop_change_angle_lst]
        if power_change_lst == 'None':
            self.power_change_lst = None
        else:
            power_change_lst = power_change_lst.split(',')
            self.power_change_lst = [float(i) for i in power_change_lst]


    def action_to_command(self, time, state, action_code):
        """
        given the index outputted by a neural network, the propeller change and power change is encoded from that.
        Those changes are returned so the caller can convert the agents network raw outputs into a change in
        control actuators. This is used for discrete actions and is expected to be used from a softmax output layer

        :param time: the time stamp of the simulation
        :param state: the current state of the boat
        :param action_code: index from neural network
        :return: propeller angle change [rad], and power change [watt]
        """

        # convert the code to a propeller change and a power change

        if self.power_change_lst is None:
            # no power change, only the propeller is changeing
            power_change = 0.0

            # get the propeller angle change
            propeller_change = self.prop_change_angle_lst[action_code]
        else:
            # control both the propeller and the power
            # proppeller angle uses column idx, power uses row idx
            prop_idx = int(action_code%len(self.prop_change_angle_lst))
            power_idx = int(action_code/len(self.power_change_lst))

            # use the indeices to get the power and propeller values
            propeller_change = self.prop_change_angle_lst[prop_idx]
            power_change = self.power_change_lst[power_idx]

        return propeller_change, power_change

    def get_action_size(self):
        """
        gives the number of actions a neural network will need to have for the action selection.

        :return: an integer for the number action slots a neural network should have.
        """
        if self.power_change_lst is None:
            # agent is only controlling the propeller angle, and not the power amount
            action_size = len(self.prop_change_angle_lst)
        else:
            # the agent is controlling both the propeller angle and the power delivered to the propeller
            action_size = len(self.prop_change_angle_lst) - len(self.power_change_lst)

        return action_size


class DirectControlActionContinous(ActionOperation):

    def __init__(self, name, max_propeller, max_power):
        """
        is a class that converts a neural networks outputs to a change in a boats propeller angle and power value

        :param name: a string of the action used to help with debuging
        :param max_propeller: the maximum turning value of the propeller [rad/s]
        :param max_power: the maximum power change value of the propeller [watt/s]
        """
        super().__init__(name)

        self.max_propeller = max_propeller
        if max_power == 'None':
            self.max_power = None
        else:
            self.max_power = max_power

    def action_to_command(self, time, state, input):
        """
        converts the outputs of a neural network to a propeller and power change amount. This is assumed a continues
        value. The input should be bounded from -1 to 1. The conversion does not have to be linear but currently
        it is linear.

        :param time: the time stamp of the simulation
        :param state: the current state of the boat
        :param input: the output of the neural networks. This is a two element vector having the normalized propeller
            change command and then the normalized power change command, or a one element vector with the normalized
            propeller change
        :return: the propeller change [rad] and the power change [watt]
        """

        propeller = input[0]
        propeller_change = propeller * self.max_propeller

        power_change = 0.0
        if len(input == 2):
            power = input[1]
            power_change = power * self.max_power

        return propeller_change, power_change

    def get_action_size(self):
        """
        tells the agent how many outputs a neural network needs to be compatable with this action formulation

        :return: integer of the action size
        """
        if self.max_power == None:
            # only controlling the propeller angle
            action_size = 1
        else:
            # controlling both the propeller and power
            action_size = 2

        return action_size


class PathActionOperation(ActionOperation):

    def __init__(self, name,replan_rate):
        """
        a super class that all action designations that use paths to navigate inherit from. The two options that
        are currently supported are direct control and path based. This forces all children to build paths

        :param name: a string to denote what action this is. This is helpful for debugging and logging
        :param replan_rate: the time in seconds the agent follows a path before replanning a new path
        """

        super().__init__(name)
        self.path = None  # the path object that has the current plan
        self.last_replan = None  # the time stamp of the last time it was replanned
        self.replan_rate = replan_rate  # the rate at which to regenrate a new path

    @abstractmethod
    def build_path(self, control_points, n_path_points):
        """
        Given a list of control points, generate a bezier spline from said control points. The generated path is what
        the boat is working to follow.

        :param control_points:
        :return:
        """


class PathContinousCp(PathActionOperation):

    def __init__(self, name, replan_rate,controller, angle_range, power_range=None, num_control_point=4):
        """
        creates paths a regular intervals by a number of control points. The control points are used to generate a
        path as a b spline. Each control point is relative from the last segments vector

        :param name: a string to help delineate what action it is. This is mainly used for debugging and logging
        :param replan_rate: the time to use a path before replanning a new path [s]
        :param controller: the controller object that helps the boat stay on the path
        :param angle_range: a list of min and max values that the agent can choose a control point from
        :param power_range: a list of min and max values for the power change at each control point
        :param num_control_point: the number of control points used to generate the path. The default number is four
            control points.
        """
        super().__init__(name,replan_rate)

        self.controller = controller  # controller object to keep the agent on the path
        angle_range = angle_range.split(',')
        self.angle_range = [float(i) for i in angle_range]

        if power_range == 'None':
            # set power to None, so that only the propeller angle is changed.
            self.power_range = None
        else:
            # both the propeller angle and power are controlled by the agent
            power_range = power_range.split(',')
            self.power_range = [float(i) for i in power_range]

        self.num_control_point = num_control_point  # number of control points to build the path from

    def build_path(self, control_points, n_path_points=16):
        """
        Given a list of control points, generate a bezier spline from said control points. The generated path is what
        the boat is working to follow.

        :param control_points:
        :return:
        """
        # generates a path in the form of (x,y) points
        self.path = bezier_curve(control_points, n_path_points)

    def action_to_command(self, time, state, input):
        """
        converts the raw output of the agent (neural network) and using the path to get the propeller angle and power
        change

        :param time: current time of the simulation
        :param state: dictionary that has all of the current state information
        :param input: raw output from the agent to be inputed into the action operation
        :return:
        """
        if (time - self.last_replan) > self.replan_rate:
            # enough time to replanc
            pass

    def get_action_size(self):
        """
        produce the number of unique actions the current action formulation can have. Here the actions are the
        relative angle, and potentially power change at each control point. This is used to build the
        neural network output layer size.
        :return:
        """
        if self.power_range is None:
            # only changing the path not the power setting along the path
            action_size = self.num_control_point
        else:
            # change the power and the path
            action_size = self.num_control_point*2

        return action_size


class PathDiscreteCp(PathActionOperation):

    def __init__(self, name, replan_rate, controller, angle_adj_lst,  power_change_lst=None,num_control_point=4):
        """
        an action type that at intervals creates a path for the agent to follow. The path is built from control points.
        This action uses a set of control point changes instead of a fully continous selection.

        :param name: a string to help delineate what action it is. This is mainly used for debugging and logging
        :param replan_rate: the time to use a path before replanning a new path [s]
        :param controller: the controller object that helps the boat stay on the path
        :param angle_adj_lst: a list of strings that list out the angles that can be
        :param power_change_lst: a list of strings that denote what power changes values are okay
        :param num_control_point: the number of control points to build the path
        """
        super().__init__(name, replan_rate)
        self.controller = controller
        angle_adj_lst = angle_adj_lst.split(',')
        self.angle_adj_lst = [float(i) for i in angle_adj_lst]
        if power_change_lst == 'None':
            self.power_change_lst = None
        else:
            power_change_lst = power_change_lst.split(',')
            self.power_change_lst = [float(i) for i in power_change_lst]
        self.num_control_point = num_control_point

    def build_path(self, control_points, n_path_points=16):
        """
        Given a list of control points, generate a bezier spline from said control points. The generated path is what
        the boat is working to follow.

        :param control_points:
        :return:
        """
        # generates a path in the form of (x,y) points
        self.path = bezier_curve(control_points, n_path_points)

    def action_to_command(self, time, state, input):
        """
        takes the raw ouput of the agent (neural network) and converts it to propeller angle change and power change
        values

        :param time: the current time of the simulation
        :param state: a dictionary that has the current state information
        :param input: the raw output from the agent (neural network) to be converted into propeller angle and power
            changes
        :return:
        """
        if (time - self.last_replan) > self.replan_rate:
            # enough time to replanc
            pass

    def get_action_size(self):
        """
        produce the number of unique actions the current action formulation can have. This is used to build the
        neural network output layer size.
        :return:
        """
        if self.power_change_lst is None:
            # only changing the path not the power setting along the path
            action_size = np.power(len(self.angle_adj_lst),self.num_control_point)
        else:
            # change the power and the path
            action_size = np.power(len(self.angle_adj_lst)*len(self.power_change_lst),self.num_control_point)

        return action_size


class PathDiscreteCanned(PathActionOperation):

    def __init__(self, name, replan_rate,  controller,turn_amount_lst, s_curve_lst, power_change_lst=None):
        """
        generates a path for the agent to follow at fixed intervals. The paths are built from two ways, s curves and
        turns. These are converted from the raw input to an encoding to select the correct path. A controller is what
        keeps the boat on the path

        :param name: a string to help delineate what action it is. This is mainly used for debugging and logging
        :param replan_rate: the time to use a path before replanning a new path [s]
        :param controller: the controller object that helps the boat stay on the path
        :param turn_amount_lst:
        :param s_curve_lst:
        :param power_change_lst:
        """
        super().__init__(name, replan_rate)
        self.controller = controller  # controller object to drive the propeller angle and power
        turn_amount_lst = turn_amount_lst.split(',')
        self.turn_amount_lst = [float(i) for i in turn_amount_lst]
        s_curve_lst = s_curve_lst.split(',')
        self.s_curve_lst = [float(i) for i in s_curve_lst]
        if power_change_lst == 'None':
            # agent is not controlling power, only propeller angle
            self.power_change_lst = None
        else:
            # agent is controlling both power and propeller angle
            self.power_change_lst = power_change_lst

    def build_path(self, control_points, n_path_points=16):
        """
        Given a list of control points, generate a bezier spline from said control points. The generated path is what
        the boat is working to follow.

        :param control_points:
        :return:
        """
        # generates a path in the form of (x,y) points
        self.path = bezier_curve(control_points, n_path_points)

    def action_to_command(self, time, state, input):
        """
        given the raw output of the agent (neural network) a path is generated if enough time has elapsed since the last
        time a path has been generated. Internally, a controller uses the path to generate propeller angle and power
        changes. that is what is returned.

        :param time: the time of the simulation
        :param state: a dictionary for the state of the simulation
        :param input: the raw output of the agent (neural network) to be converted.
        :return: a propeller angle  change [rad] and power change [watt] of the propeller
        """
        if (time - self.last_replan) > self.replan_rate:
            # enough time to replanc
            pass

    def get_action_size(self):
        """
        gets the number of action outputs the agent (neural network) needs to be compatible with this action
        formulation

        :return: integer for the number of action outputs required by the user.
        """
        if self.power_change_lst is None:
            # only changing the path not the power setting along the path
            action_size = len(self.turn_amount_lst) + len(self.s_curve_lst)
        else:
            # change the power and the path
            action_size = (len(self.turn_amount_lst) + len(self.s_curve_lst))*len(self.power_change_lst)

        return action_size