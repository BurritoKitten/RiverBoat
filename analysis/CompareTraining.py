"""
Script that grabs multiple training scenario data pieces for no obstacle cases and compares there learning performance
with respect to different information.
EX:
    - number of episodes
    - number of data points seen
    - number of data points the neural network has seen

"""



if __name__ == '__main__':

    # -----------------------------
    # edit this section

    # add a dictionary that describes the training set one wants use and add it to the total list of training scenarios
    experiments_to_compare = []

    exp_1 = {"trial_group" :'DebuggingPathDDPG', "trial_number" : 11, "label": "DDPG_Path_Control"}
    experiments_to_compare.append(exp_1)

    exp_2 = {"trial_group" :'DebuggingDirectControlDDPG', "trial_number" : 8, "label": "DDPG_Direct_Control"}
    experiments_to_compare.append(exp_2)

    exp_3 = {"trial_group": 'DebuggingDirectControlDQN', "trial_number": 8, "label": "DQN_Direct_Control"}
    experiments_to_compare.append(exp_3)

    exp_4 = {"trial_group": 'DebuggingPath', "trial_number": 8, "label": "DQN_Path_Control"}
    experiments_to_compare.append(exp_4)

    # edit this section
    # -----------------------------