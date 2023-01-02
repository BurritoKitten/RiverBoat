"""
Class that stores the experience tuples captured by the agent during training.

"""

# native modules
from collections import namedtuple
import random

# 3rd party modules
import numpy as np
import torch

# own modules

def convert_numpy_to_tensor(device, arr):
    """
    converts a numpy array to a tensor for use with pytorch
    :param arr:
    :return:
    """
    tmp = torch.tensor([arr], device=device, dtype=torch.float)
    return tmp.view(tmp.size(), -1)


class ReplayStorage:

    def __init__(self, capacity, extra_fields, strategy):
        """


        :param h_params:
        :param transition:
        """

        # save the hyperparameters
        #self.h_params = h_params
        self.strategy = strategy
        self.transition = None
        self.set_transition(extra_fields)
        self.capacity = capacity
        self.position = dict()

        # initialize the replay buffers
        self.buffers = dict()
        self.interim_buffer = None
        self.strategy_initializer(self.transition)

    def set_transition(self, extra_fields):
        """
        for the configuration outlined in the input hyper-parameter files, determine the appropriate transition data to
        be stored during each step of the simulation

        :return:
        """

        if self.strategy == 'all_in_one':
            nominal = ['state', 'action', 'next_state', 'reward', 'done']+extra_fields
            self.transition = namedtuple('Transition', tuple(nominal))
        elif self.strategy == 'proximity':
            nominal = ['state', 'action', 'next_state', 'reward', 'done','prox']+extra_fields
            self.transition = namedtuple('Transition', tuple(nominal))
        elif self.strategy == 'outcome':
            nominal = ['state', 'action', 'next_state', 'reward', 'done','outcome']+extra_fields
            self.transition = namedtuple('Transition', tuple(nominal))
        else:
            raise ValueError('Invalid option given for replay strategy. Only \'all_in_one\', \'proximity\', and \'outcome\' are allowed')

    def strategy_initializer(self, transition):
        """
        intializes the replay buffer strategy, creating the buffers based on how the memory will be stored and
        organized. Currently there exists three strategies.
            1 - All data in one buffer
            2 - data split into close to an obstacle, and data far from an obstacle
            3 - data split into tree cases: a successful episode, and episode where the boat crashes, and an episode
                where the boat does not crash or succeed

        :param transition: named tuple that defines the data that is stored in the replay buffers
        :return:
        """

        # initialize the position for the interim buffer
        self.position['interim'] = 0

        # set up the replay storage based on the strategt passed in by the hyper parameters
        if self.strategy == 'all_in_one':
            # all data samples are saved into one replay buffer
            tmp_buffer = ReplayMemory(self.capacity,transition,'only')
            self.buffers['only'] = tmp_buffer

            # initialize the position for the only buffer
            self.position['only'] = 0

        elif self.strategy == 'proximity':
            # data is saved into two buffers based on how close the boat is to an obstacle
            close_buffer = ReplayMemory(self.capacity,transition,'close')
            far_buffer = ReplayMemory(self.capacity,transition,'far')
            self.buffers['close'] = close_buffer
            self.buffers['far'] = far_buffer

            # initialize the position for the both buffers
            self.position['close'] = 0
            self.position['far'] = 0

        elif self.strategy == 'outcome':
            # data is saved into three replay buffers based on the outcome of the episode: crash, success, or others
            crash_buffer = ReplayMemory(self.capacity,transition,'crash')
            success_buffer = ReplayMemory(self.capacity,transition,'success')
            other_buffer = ReplayMemory(self.capacity,transition, 'other')
            self.buffers['crash'] = crash_buffer
            self.buffers['success'] = success_buffer
            self.buffers['other'] = other_buffer

            # initialize the position for the all buffers
            self.position['crash'] = 0
            self.position['success'] = 0
            self.position['other'] = 0

        else:
            raise ValueError('An invalid learning agent replay strategy was given, only \'all_in_one\', \'proximity\', or \'outcome\' are accepted')

        # initialize the buffer that is used during an episode to store data tuples before being sorted into the correct
        # replay buffer
        self.reset_interim_buffer()

    def push(self, *args):
        """
        pushes a single data point (a state, action, reward, next state, and reward) tuple to the interim replay buffer.
        The interim buffers keeps the data until it can be sorted

        :param transition:
        :return:
        """
        self.interim_buffer.push(*args)

    def sort_data_into_buffers(self):
        """
        The data that has been accumulated over the course of an episode is sorted into the correct replay buffers
        based on the storage strategy being used. It is assumed this function is called after the completion of an
        episode.

        :return:
        """

        if self.strategy == 'all_in_one':
            for data in self.interim_buffer.memory:
                if len(self.buffers['only']) < self.capacity:
                    self.buffers['only'].memory.append(None)
                self.buffers['only'].memory[self.position['only']] = data
                self.position['only'] = (self.position['only'] + 1) % self.capacity
        elif self.strategy == 'proximity':

            # get the proximity threshold from the hyper-parameters
            prox_thresh = self.h_params['replay_data']['proximity_threshold']

            # iterate through the data to sort it into the correct buffer
            for data in self.interim_buffer.memory:

                # determine what buffer the data should be in
                if data.prox <= prox_thresh:
                    tag = 'close'
                else:
                    tag = 'far'

                # add the data point to the buffer
                buffer = self.buffers[tag].memory
                if len(buffer) < self.capacity:
                    buffer.append(None)
                buffer[self.position[tag]] = data
                self.position[tag] = (self.position[tag] + 1) % self.capacity

        elif self.strategy == 'outcome':

            # get the last data point to determine the outcome
            tag = self.interim_buffer.memory[-1].outcome
            # iterate through the data to sort it into the correct buffer

            for data in self.interim_buffer.memory:

                buffer = self.buffers[tag].memory

                # add the data point to the buffer
                if len(buffer) < self.capacity:
                    buffer.append(None)
                buffer[self.position[tag]] = data
                self.position[tag] = (self.position[tag] + 1) % self.capacity

        self.reset_interim_buffer()

    def reset_interim_buffer(self):
        """
        set the buffer that is used to store the data during an episode to empty to be prepared for the next episode

        :return:
        """

        # create a new buffer to 'reset' the buffer
        self.interim_buffer = ReplayMemory(self.capacity,self.transition,'interim')

        # reset the position of the interim buffer
        self.position['interim'] = 0

    def reset_all_buffers(self):
        """
        resets all of the buffers to empty

        :return:
        """
        # remove all data form each buffer
        self.strategy_initializer(self.transition)

    def sample(self, batch_size):
        """
        given a request for the number of samples, a batch of data is taken from the replay buffers based on the
        strategy. The distributions from each buffer are specified in the input file.

        :param batch_size: the number of data tuples tp sample from the replay buffers to train over
        :return:
        """
        #strategy = self.h_params['replay_data']['replay_strategy']
        if self.strategy == 'all_in_one':

            if len(self.buffers['only'].memory) < batch_size:
                # not enough data to fill up one batch
                return None

            return random.sample(self.buffers['only'].memory, batch_size)
        elif self.strategy == 'proximity':

            if len(self.buffers['close'].memory) + len(self.buffers['far'].memory) < batch_size:
                # there is not enough data in both buffers to complete one batch
                return None

            n_close = int(np.floor(self.h_params['replay_data']['close_fraction']*float(batch_size)))
            n_far = batch_size - n_close

            enough_close = False
            if n_close <= len(self.buffers['close'].memory):
                enough_close = True
            enough_far = False
            if n_far <= len(self.buffers['far'].memory):
                enough_far = True

            if enough_close and enough_far:

                close = random.sample(self.buffers['close'].memory, n_close)
                far = random.sample(self.buffers['far'].memory, n_far)
            else:
                # one of the buffers does not have enough data so some needs to be borrowed from the other buffer to
                # fill out the buffer
                # sort the buffers in descending error
                buffer_lens = [['far', len(self.buffers['far'].memory), n_far],
                               ['close', len(self.buffers['close'].memory), n_close]]

                # bubble sort
                n = len(buffer_lens)
                for i in range(n):
                    for j in range(0, n - i - 1):

                        # traverse the array from 0 to n-i-1
                        # Swap if the element found is greater
                        # than the next element
                        if buffer_lens[j][1] < buffer_lens[j + 1][1]:
                            buffer_lens[j], buffer_lens[j + 1] = buffer_lens[j + 1], buffer_lens[j]

                # determine how many extra data points are needed.
                extra_needed = 0
                while len(buffer_lens) > 0:

                    tmp_buffer_info = buffer_lens.pop()

                    if len(buffer_lens) == 0:
                        adj = extra_needed
                    else:
                        adj = int(np.floor(extra_needed / len(buffer_lens)))

                    tmp_additional_data = (tmp_buffer_info[2] + adj) - len(self.buffers[tmp_buffer_info[0]].memory)
                    if tmp_additional_data > 0:
                        # there is not enough data in this buffer to meet the requested amount, so log that more
                        # data is needed from the remaining buffers
                        extra_needed = tmp_additional_data

                        if tmp_buffer_info[0] == 'far':
                            far = random.sample(self.buffers[tmp_buffer_info[0]].memory,
                                                    len(self.buffers[tmp_buffer_info[0]]))
                        elif tmp_buffer_info[0] == 'close':
                            close = random.sample(self.buffers[tmp_buffer_info[0]].memory,
                                                  len(self.buffers[tmp_buffer_info[0]]))
                    else:
                        if tmp_buffer_info[0] == 'far':
                            far = random.sample(self.buffers[tmp_buffer_info[0]].memory,
                                                    tmp_buffer_info[2] + extra_needed)
                        elif tmp_buffer_info[0] == 'close':
                            close = random.sample(self.buffers[tmp_buffer_info[0]].memory,
                                                  tmp_buffer_info[2] + extra_needed)

            batch = close + far
            return batch

        elif self.strategy == 'outcome':

            if len(self.buffers['success'].memory)+len(self.buffers['crash'].memory)+len(self.buffers['other'].memory) < batch_size:
                # there is not enough data in all of the buffers to complete one batch
                return None

            n_success = int(np.floor(self.h_params['replay_data']['success_fraction']*float(batch_size)))
            n_crash = int(np.floor(self.h_params['replay_data']['crash_fraction']*float(batch_size)))
            n_other = batch_size - n_crash - n_success

            enough_success = False
            if n_success <= len(self.buffers['success'].memory):
                enough_success = True
            enough_crash = False
            if n_crash <= len(self.buffers['crash'].memory):
                enough_crash = True
            enough_other = False
            if n_other <= len(self.buffers['other'].memory):
                enough_other = True

            if enough_success and enough_crash and enough_other:
                # there exists enough data in each buffer to meet the desired batch size in the desired ratios
                success = random.sample(self.buffers['success'].memory, n_success)
                crash = random.sample(self.buffers['crash'].memory, n_crash)
                other = random.sample(self.buffers['other'].memory, n_other)
            else:
                # one or more of the buffers does not have enough data to meet the desired ratio, but enough data exists
                # to fill up the batch. Data is pulled from the buffers that have excess to allow for training to begin
                # before each buffer has enough

                # sort the buffers in descending error
                buffer_lens = [['success',len(self.buffers['success'].memory), n_success],
                               ['crash',len(self.buffers['crash'].memory), n_crash],
                               ['other',len(self.buffers['other'].memory), n_other]]

                # bubble sort
                n = len(buffer_lens)
                for i in range(n):
                    for j in range(0, n - i - 1):

                        # traverse the array from 0 to n-i-1
                        # Swap if the element found is greater
                        # than the next element
                        if buffer_lens[j][1] < buffer_lens[j + 1][1]:
                            buffer_lens[j], buffer_lens[j + 1] = buffer_lens[j + 1], buffer_lens[j]

                # determine how many extra data points are needed.
                extra_needed = 0
                while len(buffer_lens) > 0:

                    tmp_buffer_info = buffer_lens.pop()

                    if len(buffer_lens) == 0:
                        adj = extra_needed
                    else:
                        adj = int(np.floor(extra_needed/len(buffer_lens)))

                    tmp_additional_data = (tmp_buffer_info[2]+adj)-len(self.buffers[tmp_buffer_info[0]].memory)
                    if tmp_additional_data > 0:
                        # there is not enough data in this buffer to meet the requested amount, so log that more
                        # data is needed from the remaining buffers
                        extra_needed = tmp_additional_data

                        if tmp_buffer_info[0] == 'success':
                            success = random.sample(self.buffers[tmp_buffer_info[0]].memory,len(self.buffers[tmp_buffer_info[0]]))
                        elif tmp_buffer_info[0] == 'crash':
                            crash = random.sample(self.buffers[tmp_buffer_info[0]].memory,len(self.buffers[tmp_buffer_info[0]]))
                        elif tmp_buffer_info[0] == 'other':
                            other = random.sample(self.buffers[tmp_buffer_info[0]].memory,len(self.buffers[tmp_buffer_info[0]]))
                    else:
                        if tmp_buffer_info[0] == 'success':
                            success = random.sample(self.buffers[tmp_buffer_info[0]].memory,tmp_buffer_info[2]+extra_needed)
                        elif tmp_buffer_info[0] == 'crash':
                            crash = random.sample(self.buffers[tmp_buffer_info[0]].memory,tmp_buffer_info[2]+extra_needed)
                        elif tmp_buffer_info[0] == 'other':
                            other = random.sample(self.buffers[tmp_buffer_info[0]].memory,tmp_buffer_info[2]+extra_needed)


            batch = success + crash + other
            return batch


class ReplayMemory(object):

    def __init__(self, capacity, transition, name):
        """
        A buffer that holds transition tuples that is blind to what data it stores

        :param capacity: the amount of transition tuples in the buffer
        :param transition: the tuple describing the transition tuples
        :param name: the name of the buffer the delineate between multiple buffers
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0  # position of insertion in the buffer
        self.transition = transition
        self.name = name

    def push(self, *args):
        """
        Saves a transition
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)