
# native packages

# 3rd party packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# own packages
import src.Optimizer as Optimizer
import src.ReplayMemory as ReplayMemory


class Network(nn.Module):

    def __init__(self, action_size, activation, h_params, last_activation, layer_numbers, loss, state_size, device):
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
        '''
        # build the input and hidden layers
        self.layers = []
        for i, layer in enumerate(layer_numbers):

            if i == 0:
                tmp_layer = nn.Linear(state_size, int(layer), device=device)
            else:
                tmp_layer = nn.Linear(int(layer_numbers[i-1]), int(layer), device=device)

            self.layers.append(tmp_layer)

        # get the output layer
        self.out = nn.Linear(int(layer_numbers[-1]),action_size, device=device)

        # interim activation
        hidden_active = h_params['learning_algorithm']['activation']
        if hidden_active == 'leaky_relu':
            self.active = torch.nn.LeakyReLU()
        elif hidden_active == 'linear':
            self.active = torch.nn.Linear(device=device)
        elif hidden_active == 'relu':
            self.active = torch.nn.ReLU( )
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
        '''
        self.fc1 = nn.Linear(state_size, 100, device=device)
        self.fc2 = nn.Linear(100,100, device=device)
        self.out = nn.Linear(100, action_size, device=device)
        self.relu = torch.nn.ReLU()
        #self.drop = nn.Dropout(h_params['learning_agent']['drop_out'])

    def forward(self, z):
        """
        Override the forward function to generate an ouput from the network

        :param x: input to the network. Should be a state vector that is normalized
        :return: the output of the network
        """

        """
        for layer in self.layers:
            x = layer(x)
            x = self.active(x)
            x = self.drop(x)

        x = self.out(x)

        # if not a linear output
        if self.last_active is not None:
            x = self.last_active(x)
        """

        x = self.fc1(z)
        x = self.relu(x)
        x = self.fc2(x)
        #x = self.drop(x)
        x = self.relu(x)
        x = self.out(x)

        return x

    def getLoss(self):
        pass


class LearningAlgorithms:

    def __init__(self, action_size, activation, h_params, last_activation, layer_numbers, loss, state_size, n_batches, batch_size, device):
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
        :param n_batches: the number of batches to train the neural networks
        :param batch_size: the amount of data in a batch
        :param device: either the cpu or cuda for the gpu
        :return:
        """

        self.action_size = action_size
        self.activation = activation
        self.h_params = h_params
        self.last_activation = last_activation
        self.layer_numbers = layer_numbers
        self.loss = loss
        self.state_size = state_size
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.gamma = h_params['learning_algorithm']['gamma']
        self.device = device
        self.optimizer = None

        self.output_history = []

    def reset_output_history(self):
        """
        Reset stored values from the output of the network
        :return:
        """
        self.output_history = []


class DQN(LearningAlgorithms):

    def __init__(self, action_size, activation, h_params, last_activation, layer_numbers, loss, state_size, n_batches, batch_size, device, optimizer_settings):
        """
        The agent uses a deep Q network for the agent.

        :param action_size: the number of action outputs that agent needs
        :param activation: the activation function for the first and deep layers
        :param last_activation: the activation function for the last layer
        :param layer_numbers: The number of internal layers of the neural network
        :param loss: a string designating the loss function
        :param state_size: the number of inputs for the neural network
        :param n_batches: the number of batches to train the neural networks
        :param batch_size: the amount of data in a batch
        :param device: either the cpu or cuda for the gpu
        :param optimizer: the optimizer used to improve the weights of the network
        :return:
        """
        super().__init__(action_size, activation, h_params, last_activation, layer_numbers, loss, state_size, n_batches, batch_size, device)

        self.name = 'DQN'

        # create the policy network
        self.network = Network(action_size, activation, h_params, last_activation, layer_numbers, loss, state_size, device)

        # create the target network
        self.target_network = Network(action_size, activation, h_params, last_activation, layer_numbers, loss, state_size, device)

        self.optimizer = Optimizer.get_optimizer(self.network.parameters(), optimizer_settings)

    def get_output(self, inp, is_grad=False):
        """
        get the output from the network

        :param inp: input list. Should be the observations from the simulation
        :return:
        """
        inp = ReplayMemory.convert_numpy_to_tensor(self.device, inp)
        if is_grad:
            out = self.network.forward(torch.Tensor(inp))
            #self.output_history.append(out.to('cpu').numpy()[0])
            return out
        else:
            with torch.no_grad():
                out = self.network.forward(torch.Tensor(inp))
                self.output_history.append(out.to('cpu').numpy()[0])
                return out

    def train_agent(self, replay_storage):
        """
        train the networks with the available data
        :return:
        """

        loss = None
        c = 0
        while c < self.n_batches:

            transitions = replay_storage.sample(self.batch_size)
            if transitions is None:
                return

            batch = replay_storage.transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None])

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # use the policy network
            #state_action_values = self.network(state_batch).gather(1, action_batch.type(torch.int64))
            #tmp_out = self.network(state_batch)
            #tmp_2 = self.network(state_batch).gather(1, action_batch.type(torch.int64))
            #state_action_values = torch.gather(tmp_out,1,action_batch.type(torch.int64))
            state_action_values = self.network(state_batch).gather(1, action_batch.type(torch.int64))

            #next_state_values = torch.zeros(self.h_params['learning_agent']['batch_size'], device=self.device)
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            #test = torch.zeros(self.batch_size, device=self.device)
            #test[non_final_mask] = self.network(non_final_next_states).max(1)[0].detach()
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()

            expected_state_action_values = torch.add(
                torch.mul(torch.reshape(next_state_values, (len(next_state_values), 1)),
                          self.gamma), reward_batch)

            if self.loss == 'huber':
                loss = F.smooth_l1_loss(state_action_values.type(torch.double),
                                        expected_state_action_values.type(torch.double))
            elif self.loss == 'mse':
                loss = F.mse_loss(state_action_values.type(torch.double),
                                  expected_state_action_values.type(torch.double))
            else:
                raise ValueError('Given loss is currently not supported')

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.network.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            c += 1

        return loss

    def update_target_network(self):
        """
        copies the values of the policy network into the target network
        :return:
        """
        self.target_network.load_state_dict(self.network.state_dict())

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