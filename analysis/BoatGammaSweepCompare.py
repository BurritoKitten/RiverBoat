"""
graphs the success rate for different discount factors for the riverboat
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def graphComparison(exp_groups):

    sns.set_theme()
    fig1 = plt.figure(1)#,figsize=(14,8))
    ax1_1 = fig1.add_subplot(111)
    fig2 = plt.figure(2)#, figsize=(14, 8))
    ax2_1 = fig2.add_subplot(111)
    fig3 = plt.figure(3)#, figsize=(14, 8))
    ax3_1 = fig3.add_subplot(111)
    fig4 = plt.figure(4)#, figsize=(14, 8))
    ax4_1 = fig4.add_subplot(111)

    # loop over the gamma values
    for key, group in exp_groups.items():

        # loop over the scenarios to average
        avg_min_dst = []
        avg_cum_reward = []
        avg_sim_time = []
        success_rate = []
        for i, exp_details in enumerate(group):
            tmp_df = pd.read_csv('../Output/'+str(exp_details['trial_group'])+'/'+str(exp_details['trial_number'])+'/Progress/Data/evaluation_average.csv')
            tmp_success = tmp_df['successRate'].values
            if success_rate == []:
                epNum = tmp_df['EpNum'].values
                success_rate = np.zeros((len(tmp_success), len(group)))
                avg_min_dst = np.zeros_like(success_rate)
                avg_cum_reward = np.zeros_like(success_rate)
                avg_sim_time = np.zeros_like(success_rate)
            success_rate[:,i] = tmp_success
            avg_min_dst[:,i] = tmp_df['avgMinDst']
            avg_cum_reward[:,i] = tmp_df['avgCumReward']
            avg_sim_time[:,i] = tmp_df['simTime']


        # get the row wize average to get the mean as a function of training progress.
        mean_success_rate = np.mean(success_rate,axis=1)
        mean_min_dst = np.mean(avg_min_dst,axis=1)
        mean_cum_reward = np.mean(avg_cum_reward, axis=1)
        mean_sim_time = np.mean(avg_sim_time, axis=1)

        ax1_1.plot(epNum,mean_success_rate,label="$\gamma="+str(key)+"$")
        ax2_1.plot(epNum, mean_min_dst, label="$\gamma=" + str(key) + "$")
        ax3_1.plot(epNum, mean_cum_reward, label="$\gamma=" + str(key) + "$")
        ax4_1.plot(epNum, mean_sim_time, label="$\gamma=" + str(key) + "$")

    ax1_1.set_xlabel('Episode Number [-]')
    ax1_1.set_ylabel('Evaluation Set Success Rate [-]')
    ax1_1.legend()
    fig1.suptitle('Average Success Rate of Evaluation Set Over 3 Training Sequences')
    fig1.tight_layout()

    ax2_1.set_xlabel('Episode Number [-]')
    ax2_1.set_ylabel('Evaluation Set Average Minimum Distance To Goal [m]')
    ax2_1.legend()
    fig2.suptitle('Average Minimum Distance to Goal of Evaluation Set Over 3 Training Sequences')
    fig2.tight_layout()

    ax3_1.set_xlabel('Episode Number [-]')
    ax3_1.set_ylabel('Evaluation Set Average Cumulative Reward [-]')
    ax3_1.legend()
    fig3.suptitle('Average Cumulative Reward of Evaluation Set Over 3 Training Sequences')
    fig3.tight_layout()

    ax4_1.set_xlabel('Episode Number [-]')
    ax4_1.set_ylabel('Evaluation Set Average Simulation Length [s]')
    ax4_1.legend()
    fig4.suptitle('Average Simulation Length of Evaluation\nSet Over 3 Training Sequences')
    fig4.tight_layout()

    plt.show()


if __name__ == '__main__':
    # -----------------------------
    # edit this section

    # dictionary to hold list of experiment information. Each key is the discount factor gamma used for the set. Each
    # list associated with the key are the repeats to be averaged for the graphs.
    experiments_to_compare = dict()

    group_1 = []
    group_1.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 6})
    group_1.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 8})
    group_1.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 10})
    experiments_to_compare[0.8] = group_1

    group_2 = []
    group_2.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 5})
    group_2.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 7})
    group_2.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 9})
    experiments_to_compare[0.85] = group_2

    group_3 = []
    group_3.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 2})
    group_3.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 11})
    group_3.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 13})
    experiments_to_compare[0.9] = group_3

    group_4 = []
    group_4.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 3})
    group_4.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 12})
    group_4.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 14})
    experiments_to_compare[0.95] = group_4

    group_5 = []
    group_5.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 4})
    group_5.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 15})
    group_5.append({"trial_group": 'TuneDDPGPathControlSparseReward', "trial_number": 16})
    experiments_to_compare[0.99] = group_5

    # edit this section
    # -----------------------------
    graphComparison(experiments_to_compare)
