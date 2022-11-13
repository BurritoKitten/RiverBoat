
# native packages

# 3rd party packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# own packages

class Network(nn.Module):

    def __init__(self, action_size, activation, h_params, last_activation, layer_numbers, loss, state_size):
        """
        builds and creates a neural network for the learning agents to learn

        :param action_size: the number of action outputs that agent needs
        :param activation: the activation function for the first and deep layers
        :param h_params: a dictionary that has the input file information that describes how to build and run the
            simulation
        :param last_activation: the activation function for the last layer
        :param layer_numbers: The number of internal layers of the neural network
        :param loss: a string designating the loss function
        :param state_size: the number of inputs for the neural network
        """
        super(Network, self).__init__()

        # build the input and hidden layers
        self.layers = []
        for i, layer in enumerate(layer_numbers):

            if i == 0:
                tmp_layer = nn.Linear(state_size, int(layer))
            else:
                tmp_layer = nn.Linear(int(layer_numbers[i-1]), int(layer))

            self.layers.append(tmp_layer)

        # get the output layer
        self.out = nn.Linear(int(layer_numbers[-1]),action_size)

        # interim activation
        hidden_active = h_params['learning_algorithm']['activation']
        if hidden_active == 'leaky_relu':
            self.active = torch.nn.LeakyReLU()
        elif hidden_active == 'linear':
            self.active = torch.nn.Linear()
        elif hidden_active == 'relu':
            self.active = torch.nn.ReLU()
        elif hidden_active == 'tan_h':
            self.active == torch.nn.Tanh()
        else:
            raise ValueError('Activation function not currently supported')

        # last layer activation
        last_active = h_params['learning_algorithm']['last_activation']
        if last_active == 'linear':
            self.last_active = None
        elif last_active == 'softmax':
            self.last_active = torch.nn.Softmax()
        else:
            raise ValueError('Last activation function not currently supported')

        # prepare dropout layers
        self.drop = nn.Dropout(h_params['learning_algorithm']['drop_out'])

    def forward(self, x):
        """
        Override the forward function to generate an ouput from the network

        :param x: input to the network. Should be a state vector that is normalized
        :return: the output of the network
        """
        for layer in self.layers:
            x = layer(x)
            x = self.active(x)
            x = self.drop(x)

        x = self.out(x)

        # if not a linear output
        if self.last_active is not None:
            x = self.last_active(x)

        return x

    def getLoss(self):
        pass


class LearningAlgorithms:

    def __init__(self, action_size, activation, h_params, last_activation, layer_numbers, loss, state_size):
        """
        A super class for learning algorithms. learning algorithms are the core of the agent's policy. Neural networks
        are used as the function approximators

        :param action_size: the number of action outputs that agent needs
        :param activation: the activation function for the first and deep layers
        :param h_params: a dictionary for simulation configuration
        :param last_activation: the activation function for the last layer
        :param layer_numbers: The number of internal layers of the neural network
        :param loss: a string designating the loss function
        :param state_size: the number of inputs for the neural network
        :return:
        """

        self.action_size = action_size
        self.activation = activation
        self.h_params = h_params
        self.last_activation = last_activation
        self.layer_numbers = layer_numbers
        self.loss = loss
        self.state_size = state_size


class DQN(LearningAlgorithms):

    def __init__(self, action_size, activation, h_params, last_activation, layer_numbers, loss, state_size):
        """
        The agent uses a deep Q network for the agent.

        :param action_size: the number of action outputs that agent needs
        :param activation: the activation function for the first and deep layers
        :param last_activation: the activation function for the last layer
        :param layer_numbers: The number of internal layers of the neural network
        :param loss: a string designating the loss function
        :param state_size: the number of inputs for the neural network
        :return:
        """
        super().__init__(action_size, activation, h_params, last_activation, layer_numbers, loss, state_size)

        self.name = 'DQN'

        # create the policy network
        self.network = Network(action_size, activation, h_params, last_activation, layer_numbers, loss, state_size)

        # create the target network
        self.target_network = Network(action_size, activation, h_params, last_activation, layer_numbers, loss, state_size)

    def get_output(self, inp, is_grad=False):
        """
        get the output from the network

        :param inp: input list. Should be the observations from the simulation
        :return:
        """
        if is_grad:
            return self.network.forward(torch.Tensor(inp))
        else:
            with torch.no_grad():
                return self.network.forward(torch.Tensor(inp))

class DDPG(LearningAlgorithms):

    def __init__(self, action_size, activation, last_activation, layer_numbers, loss, state_size):
        """
        The agent uses DDPG for the agent

        :param action_size: the number of action outputs that agent needs
        :param activation: the activation function for the first and deep layers
        :param last_activation: the activation function for the last layer
        :param layer_numbers: The number of internal layers of the neural network
        :param loss: a string designating the loss function
        :param state_size: the number of inputs for the neural network
        :return:
        """
        super().__init__(action_size, activation, last_activation, layer_numbers, loss, state_size)

        self.name = 'DDPG'