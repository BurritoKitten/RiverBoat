"""
Script that grabs multiple training scenario data pieces for no obstacle cases and compares there learning performance
with respect to different information.
EX:
    - number of episodes
    - number of data points seen
    - number of data points the neural network has seen

"""

# native modules

# 3rd party modules
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

# own modules

def graphComparisons(exps):

    sns.set_theme()
    fig_1 = plt.figure(0,figsize=(14,8))
    ax1_1 = fig_1.add_subplot(2,2,1)
    ax1_2 = fig_1.add_subplot(2, 2, 2)
    ax1_3 = fig_1.add_subplot(2, 2, 3)
    ax1_4 = fig_1.add_subplot(2, 2, 4)

    fig_2 = plt.figure(1, figsize=(14, 8))
    ax2_1 = fig_2.add_subplot(2,2, 1)
    ax2_2 = fig_2.add_subplot(2, 2, 2)
    ax2_3 = fig_2.add_subplot(2, 2, 3)
    ax2_4 = fig_2.add_subplot(2, 2, 4)

    fig_3 = plt.figure(2, figsize=(14, 8))
    ax3_1 = fig_3.add_subplot(2,2,1)
    ax3_2 = fig_3.add_subplot(2, 2, 2)
    ax3_3 = fig_3.add_subplot(2, 2, 3)
    ax3_4 = fig_3.add_subplot(2, 2, 4)
    #ax4 = fig.add_subplot(2, 2, 4)

    # loop over the experiments and graph each one in turn
    for exp in exps:
        file_name = "../Output/"+str(exp['trial_group'])+"/"+str(exp['trial_number'])+'/hyper_parameters.yaml'
        # get the h_param file to estimate the data rates
        with open(file_name, "r") as stream:
            try:
                hp_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # get info for data efficiency
        batch_size = hp_data['learning_algorithm']['batch_size']
        n_batches = hp_data['learning_algorithm']['n_batches']
        if hp_data['action_description']['action_designation'] == 'direct':
            # direct control and not path control
            eff_time_step = hp_data['scenario']['time_step']
        else:
            # path control
            eff_time_step = hp_data['action_description']['replan_rate']
        max_time = hp_data['scenario']['max_time']

        # comparison metrics
        samples_per_episode = max_time/eff_time_step
        samples_seen_by_network_per_episode = batch_size*n_batches

        # open results files of the evaluations
        eval_avg_df = pd.read_csv("../Output/"+str(exp['trial_group'])+"/"+str(exp['trial_number'])+"/Progress/Data/evaluation_average.csv")
        eval_df = pd.read_csv("../Output/" + str(exp['trial_group']) + "/" + str(exp['trial_number']) + "/Progress/Data/evaluation.csv")
        training_df = pd.read_csv("../Output/" + str(exp['trial_group']) + "/" + str(exp['trial_number']) + "/Progress/Data/training_progress.csv")

        # plot eval_avg with raw epsiode number
        ax1_1.plot(eval_avg_df['EpNum'].values,eval_avg_df['successRate'], label=exp['label'])
        ax1_2.plot(eval_avg_df['EpNum'].values, eval_avg_df['avgMinDst'], label=exp['label'])
        ax1_3.plot(eval_avg_df['EpNum'].values, eval_avg_df['avgCumReward'], label=exp['label'])
        ax1_4.plot(eval_avg_df['EpNum'].values, eval_avg_df['simTime'], label=exp['label'])

        # plot eval progress with respect to number of samples generated in simulation
        eval_avg_df['NumSamples'] = eval_avg_df['EpNum'].values*samples_per_episode
        ax2_1.semilogx(eval_avg_df['NumSamples'].values,eval_avg_df['successRate'], label=exp['label'])
        ax2_2.semilogx(eval_avg_df['NumSamples'].values, eval_avg_df['avgMinDst'], label=exp['label'])
        ax2_3.semilogx(eval_avg_df['NumSamples'].values, eval_avg_df['avgCumReward'], label=exp['label'])
        ax2_4.semilogx(eval_avg_df['NumSamples'].values, eval_avg_df['simTime'], label=exp['label'])

        # plot eval progress with respect to number of data points seen by the network
        eval_avg_df['NumSamples'] = eval_avg_df['EpNum'].values * samples_seen_by_network_per_episode
        ax3_1.plot(eval_avg_df['NumSamples'].values, eval_avg_df['successRate'], label=exp['label'])
        ax3_2.plot(eval_avg_df['NumSamples'].values, eval_avg_df['avgMinDst'], label=exp['label'])
        ax3_3.plot(eval_avg_df['NumSamples'].values, eval_avg_df['avgCumReward'], label=exp['label'])
        ax3_4.plot(eval_avg_df['NumSamples'].values, eval_avg_df['simTime'], label=exp['label'])

    fig_1.suptitle('Average Rate for Evaluation Episodes W.R.T. Number of Episodes')
    ax1_1.set_ylabel('Average Success Rate [-]')
    ax1_1.set_xlabel('Episode Number [-]')
    ax1_1.legend()
    ax1_2.set_ylabel('Average Minimum Distance To Goal [m]')
    ax1_2.set_xlabel('Episode Number [-]')
    ax1_2.legend()
    ax1_3.set_ylabel('Average Cumulative Reward \n(may be different functions) [-]')
    ax1_3.set_xlabel('Episode Number [-]')
    ax1_3.legend()
    ax1_4.set_ylabel('Average Time to Reach Destination (or Fail) [s]')
    ax1_4.set_xlabel('Episode Number [-]')
    ax1_4.legend()
    fig_1.tight_layout()

    fig_2.suptitle('Average Rate for Evaluation Episodes W.R.T. Number of Samples')
    ax2_1.grid(b=True, which='major', color='w', linewidth=1.0)
    ax2_1.grid(b=True, which='minor', color='w', linewidth=0.5)
    ax2_1.set_ylabel('Success Rate [-]')
    ax2_1.set_xlabel('Number of Samples Generated [-]')
    ax2_1.legend()
    ax2_2.set_ylabel('Average Minimum Distance To Goal [m]')
    ax2_2.set_xlabel('Number of Samples Generated [-]')
    ax2_2.legend()
    ax2_3.set_ylabel('Average Cumulative Reward \n(may be different functions) [-]')
    ax2_3.set_xlabel('Number of Samples Generated [-]')
    ax2_3.legend()
    ax2_4.set_ylabel('Average Time to Reach Destination (or Fail) [s]')
    ax2_4.set_xlabel('Number of Samples Generated [-]')
    ax2_4.legend()
    fig_2.tight_layout()

    fig_3.suptitle('Average Rate for Evaluation Episodes W.R.T. Number of Samples Seen by Neural Networks')
    ax3_1.set_ylabel('Success Rate [-]')
    ax3_1.set_xlabel('Number of Samples Seen By Agents [-]')
    ax3_1.legend()
    ax3_2.set_ylabel('Average Minimum Distance To Goal [m]')
    ax3_2.set_xlabel('Number of Samples Seen By Agents [-]')
    ax3_2.legend()
    ax3_3.set_ylabel('Average Cumulative Reward \n(may be different functions) [-]')
    ax3_3.set_xlabel('Number of Samples Seen By Agents [-]')
    ax3_3.legend()
    ax3_4.set_ylabel('Average Time to Reach Destination (or Fail) [s]')
    ax3_4.set_xlabel('Number of Samples Seen By Agents [-]')
    ax3_4.legend()
    fig_3.tight_layout()

    plt.show()

if __name__ == '__main__':

    # -----------------------------
    # edit this section

    # add a dictionary that describes the training set one wants use and add it to the total list of training scenarios
    experiments_to_compare = []

    exp_1 = {"trial_group" :'DebuggingPathDDPG', "trial_number" : 12, "label": "DDPG_PC"}
    experiments_to_compare.append(exp_1)

    exp_2 = {"trial_group" :'DebuggingDirectControlDDPG', "trial_number" : 9, "label": "DDPG_DC"}
    experiments_to_compare.append(exp_2)

    exp_3 = {"trial_group": 'DebuggingDirectControlDQN', "trial_number": 3, "label": "DQN_DC"}
    experiments_to_compare.append(exp_3)

    exp_4 = {"trial_group": 'DebuggingPath', "trial_number": 8, "label": "DQN_PC"}
    experiments_to_compare.append(exp_4)

    '''
    exp_5 = {"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 0, "label": "DDPG_PC_Sparse"}
    experiments_to_compare.append(exp_5)

    exp_6 = {"trial_group": 'TuneDDPGDirectControlSparseReward', "trial_number": 1,
             "label": "DDPG_DC_Sparse"}
    experiments_to_compare.append(exp_6)
    '''

    # edit this section
    # -----------------------------

    graphComparisons(experiments_to_compare)