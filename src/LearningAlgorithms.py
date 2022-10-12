

class LearningAlgorithms:

    def LeaningAlgorthings(self, action_size, activation, last_activation, layer_numbers, loss, state_size):
        """
        A super class for learning algorithms. learning algorithms are the core of the agent's policy. Neural networks
        are used as the function approximators

        :param action_size: the number of action outputs that agent needs
        :param activation: the activation function for the first and deep layers
        :param last_activation: the activation function for the last layer
        :param layer_numbers: The number of internal layers of the neural network
        :param loss: a string designating the loss function
        :param state_size: the number of inputs for the neural network
        :return:
        """

        self.action_size = action_size
        self.activation = activation
        self.last_activation = last_activation
        self.layer_numbers = layer_numbers
        self.loss = loss
        self.state_size = state_size

class DQN(LearningAlgorithms):

    def DQN(self, action_size, activation, last_activation, layer_numbers, loss, state_size):
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
        super().__init__(action_size, activation, last_activation, layer_numbers, loss, state_size)

        self.name = 'DQN'

class DDPG(LearningAlgorithms):

    def DDPG(self, action_size, activation, last_activation, layer_numbers, loss, state_size):
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