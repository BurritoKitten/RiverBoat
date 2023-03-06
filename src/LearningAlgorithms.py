
# native packages
from abc import ABC, abstractmethod

# 3rd party packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# own packages
import src.Optimizer as Optimizer
import src.ReplayMemory as ReplayMemory


class QNetwork(nn.Module):

    def __init__(self, action_size, h_params,  layer_numbers, state_size, device):
        """
        builds and creates a neural network for the learning agents to learn

        :param action_size: the number of action outputs that agent needs
        :param h_params: a dictionary that has the input file information that describes how to build and run the
            simulation
        :param layer_numbers: The number of internal layers of the neural network
        :param state_size: the number of inputs for the neural network
        :param device: cpu or cuda for what is used to train the networks
        """
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, int(layer_numbers[0]), device=device)
        self.fc2 = nn.Linear(int(layer_numbers[0]),int(layer_numbers[1]), device=device)
        self.out = nn.Linear(int(layer_numbers[1]), action_size, device=device)

        # interim activation
        if h_params['learning_algorithm']['activation'] == 'relu':
            self.active = torch.nn.ReLU()
        elif h_params['learning_algorithm']['activation'] == 'leaky_relu':
            self.active = torch.nn.LeakyReLU()
        else:
            raise ValueError('Not supported activation function')

    def forward(self, z):
        """
        Override the forward function to generate an ouput from the network

        :param x: input to the network. Should be a state vector that is normalized
        :return: the output of the network
        """


        x = self.fc1(z)
        x = self.active(x)
        x = self.fc2(x)
        x = self.active(x)
        x = self.out(x)

        return x


class ActorNetwork(nn.Module):

    def __init__(self, action_size, h_params, layer_numbers, state_size, device, max_action):
        """
        builds and creates a neural network for the learning agents to learn

        :param action_size: the number of action outputs that agent needs
        :param h_params: a dictionary that has the input file information that describes how to build and run the
            simulation
        :param layer_numbers: The number of internal layers of the neural network
        :param state_size: the number of inputs for the neural network
        :param device: cpu or cuda for what is used to train the networks
        """
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, int(layer_numbers[0]), device=device)
        self.fc2 = nn.Linear(int(layer_numbers[0]), int(layer_numbers[1]), device=device)
        self.out = nn.Linear(int(layer_numbers[1]), action_size, device=device)

        # interim activation
        self.active = torch.nn.ReLU()

        #
        self.out_active = torch.nn.Tanh()

        self.max_action = ReplayMemory.convert_numpy_to_tensor('cuda',max_action)

    def forward(self, z):
        """
        Override the forward function to generate an ouput from the network

        :param x: input to the network. Should be a state vector that is normalized
        :return: the output of the network
        """

        x = self.fc1(z)
        x = self.active(x)
        x = self.fc2(x)
        x = self.active(x)
        x = self.out(x)
        x = self.out_active(x)
        x = self.max_action * x # should be one for discrete outputs or untransformed outputs

        return x


class CriticNetwork(nn.Module):

    def __init__(self, action_size, h_params, layer_numbers, state_size, device):
        """
        builds and creates a neural network for the learning agents to learn

        :param action_size: the number of action outputs that agent needs
        :param h_params: a dictionary that has the input file information that describes how to build and run the
            simulation
        :param layer_numbers: The number of internal layers of the neural network
        :param state_size: the number of inputs for the neural network
        :param device: cpu or cuda for what is used to train the networks
        """
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size+action_size, int(layer_numbers[0]), device=device)
        self.fc2 = nn.Linear(int(layer_numbers[0]), int(layer_numbers[1]), device=device)
        self.out = nn.Linear(int(layer_numbers[1]), 1, device=device)

        # interim activation
        self.active = torch.nn.ReLU()

    def forward(self, z):
        """
        Override the forward function to generate an ouput from the network

        :param x: input to the network. Should be a state vector that is normalized
        :return: the output of the network
        """

        x = self.fc1(z)
        x = self.active(x)
        x = self.fc2(x)
        x = self.active(x)
        x = self.out(x)

        return x

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

    @abstractmethod
    def save_networks(self, ep_num, file_path):
        pass


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
        self.network = QNetwork(action_size, h_params, layer_numbers, state_size, device)

        # create the target network
        self.target_network = QNetwork(action_size, h_params, layer_numbers, state_size, device)

        # create the optimizer
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
            return out, out
        else:
            with torch.no_grad():
                out = self.network.forward(torch.Tensor(inp))
                self.output_history.append(out.to('cpu').numpy()[0])
                return out, out

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
            state_action_values = self.network(state_batch).gather(1, action_batch.type(torch.int64))

            next_state_values = torch.zeros(self.batch_size, device=self.device)
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

    def save_networks(self, episode_number, file_path):
        torch.save(self.network.state_dict(),file_path+str(episode_number)+'.pymdl')


class DDPG(LearningAlgorithms):

    def __init__(self, action_size, activation, h_params, last_activation, layer_numbers, loss, state_size, tau, n_batches, batch_size, device, optimizer_settings, max_action_val):
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
        super().__init__(action_size, activation, h_params, last_activation, layer_numbers, loss, state_size, n_batches, batch_size, device)

        self.name = 'DDPG'
        self.tau = tau

        # actor network

        self.actor_policy_net = ActorNetwork(action_size, h_params, layer_numbers, state_size, device, max_action_val)
        self.actor_target_net = ActorNetwork(action_size, h_params,  layer_numbers,  state_size, device, max_action_val)
        self.actor_target_net.load_state_dict(self.actor_policy_net.state_dict())
        self.actor_target_net.eval()

        # critic network
        self.critic_net = CriticNetwork(action_size, h_params,  layer_numbers, state_size, device)
            #Critic(state_size, action_size, h_params).cuda()
        self.critic_target_net = CriticNetwork(action_size, h_params, layer_numbers, state_size, device)
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())
        self.critic_target_net.eval()

        # create the optimizer
        self.actor_optimizer = Optimizer.get_optimizer(self.actor_policy_net.parameters(), optimizer_settings)

        self.critic_optimizer = Optimizer.get_optimizer(self.critic_net.parameters(), optimizer_settings)

    def get_output(self, inp, is_grad=False):
        """
        get the output from the network

        :param inp: input list. Should be the observations from the simulation
        :return:
        """
        #inp_org = inp
        inp_tensor = ReplayMemory.convert_numpy_to_tensor(self.device, inp)
        if is_grad:

            out = self.actor_policy_net(torch.Tensor(inp_tensor))

            with torch.no_grad():
                sa = np.concatenate([inp, out.cpu().detach().numpy()[0]])
                sa = ReplayMemory.convert_numpy_to_tensor(self.device, sa)
                critic_values = self.critic_net(sa)
            return out, critic_values
        else:
            with torch.no_grad():
                #dim = inp.dim()
                #if dim > 2:
                #    check = 0
                out = self.actor_policy_net(torch.Tensor(inp_tensor))
                #out = torch.concat((torch.Tensor(inp), action), dim=1)
                self.output_history.append(out.to('cpu').numpy()[0])

                sa = np.concatenate([inp,out.cpu().detach().numpy()[0]])
                sa = ReplayMemory.convert_numpy_to_tensor(self.device, sa)
                critic_values = self.critic_net(sa)
                return out, critic_values

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

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            next_state_batch = torch.cat(batch.next_state)
            #done_batch = torch.FloatTensor(1 - np.array(batch.done))
            done_batch = 1 - torch.cat(batch.done)

            target_Q_init = self.critic_target_net(
                torch.cat([next_state_batch, self.actor_target_net(next_state_batch)], dim=1))
            # target_Q_init = torch.reshape()
            done_batch = torch.reshape(done_batch, (len(done_batch), 1))
            target_Q = reward_batch + (done_batch * self.gamma * target_Q_init).detach()

            # inp = torch.cat([state_batch, action_batch], dim=1).to(torch.float)
            # current_Q = self.critic_net(torch.cat([state_batch, self.actor_policy_net(state_batch)], dim=1))
            # k =
            current_Q = self.critic_net(torch.cat([state_batch, action_batch], dim=1))
            if self.loss == 'huber':
                critic_loss = F.smooth_l1_loss(current_Q.type(torch.double),
                                        target_Q.type(torch.double))
            elif self.loss == 'mse':
                critic_loss = F.mse_loss(current_Q.type(torch.double),
                                  target_Q.type(torch.double))
            else:
                raise ValueError('Given loss is currently not supported')
            #critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic_net(torch.cat([state_batch, self.actor_policy_net(state_batch)], dim=1)).mean()
            self.actor_policy_net.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.update_target_network()

            c += 1

        return loss

    def update_target_network(self):
        """
        copies the values of the policy network into the target network
        :return:
        """

        for param, target_param in zip(self.critic_net.parameters(), self.critic_target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_policy_net.parameters(), self.actor_target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save_networks(self, episode_number, file_path):
        torch.save(self.actor_policy_net.state_dict(),file_path+str(episode_number)+'_Actor.pymdl')
        torch.save(self.critic_net.state_dict(), file_path + str(episode_number) + '_Critic.pymdl')