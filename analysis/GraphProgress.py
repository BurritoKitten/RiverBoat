"""
After a training is complete, this creates graphs of the training progress for interpreting how training is progressing
"""

# native modules

# 3rd party modules
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import seaborn as sns
import yaml

# own modules

def running_avg(data, window_size):

    avg = np.zeros_like(data,dtype=float)
    for i, tmp in enumerate(data):
        if i < int(window_size / 2):
            # tmp = cumulative_reward[0:window_size]
            sum_val = np.sum(data[0:window_size])
            avg_tmp = sum_val/window_size
            avg[i] = np.average(data[0:window_size])
        elif i > (len(data) - 1 - int(window_size / 2)):
            # tmp = cumulative_reward[len(cumulative_reward)-window_size:len(cumulative_reward)]
            avg[i] = np.average(data[len(data) - window_size:len(data)])
        else:
            # tmp = cumulative_reward[i-int(window_size/2):i+int(window_size/2)+1]
            avg[i] = np.average(data[i - int(window_size / 2):i + int(window_size / 2) + 1])

    return avg

def create_std_polygon_verts(x_data,y_data,window_size):
    """
    takes a list of data points in time and generates a set of verticies that have the standard deviation around the
    x_data

    :param x_data: this is the smoothed values that the std is centered around
    :param y_data: this is the raw data to calculate the std from
    :param window_size: the window over by which the std is calculated
    :return:
    """
    std = np.zeros_like(y_data)
    s = (len(y_data) - 1 - int(window_size / 2))
    for i, tmp in enumerate(y_data):
        if i < int(window_size / 2):
            # tmp = cumulative_reward[0:window_size]
            std[i] = np.std(y_data[0:window_size])
        elif i > (len(y_data) - 1 - int(window_size / 2)):
            # tmp = cumulative_reward[len(cumulative_reward)-window_size:len(cumulative_reward)]
            std[i] = np.std(y_data[len(y_data) - window_size:len(y_data)])
        else:
            # tmp = cumulative_reward[i-int(window_size/2):i+int(window_size/2)+1]
            std[i] = np.std(y_data[i - int(window_size / 2):i + int(window_size / 2) + 1])
    upper_std = x_data + std
    lower_std = x_data - std
    lower_std = list(reversed(lower_std))
    total_std = np.concatenate([upper_std, lower_std])

    poly = np.zeros([len(total_std), 2])
    poly[:, 0] = np.concatenate([df['ep_num'].values, list(reversed(df['ep_num'].values))])
    poly[:, 1] = total_std

    return poly

if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # Edit this block to control what is rendered ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # input controls.
    trial_group = 'DebuggingPathDDPG'
    trial_number = 6
    window_size = 200

    small_window = 75
    medium_window = 125
    large_window = 250
    huge_window = 500


    # ------------------------------------------------------------------------------------------------------------------
    # Edit this block to control what is rendered ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # get hyperparameters to get epsilon schedule
    file_name = '..\\Output\\'+str(trial_group)+'\\'+str(trial_number)+'\\hyper_parameters.yaml'
    with open(file_name, "r") as stream:
        try:
            hp_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    epsilon_schedule = hp_data['action_description']['eps_schedule']
    tuples = epsilon_schedule.split(';')
    epsilon_schedule = []
    for tup in tuples:
        tmp_tup = tup.split(",")
        tmp_tup[0] = int(tmp_tup[0])
        tmp_tup[1] = float(tmp_tup[1])
        epsilon_schedule.append(tmp_tup)



    file_name = '..\\Output\\'+str(trial_group)+'\\'+str(trial_number)+'\\Progress\\Data\\training_progress.csv'
    df = pd.read_csv(file_name)
    max_ep_num = df['ep_num'].max()
    if max_ep_num >epsilon_schedule[-1][0]:
        epsilon_schedule.append([max_ep_num,epsilon_schedule[-1][1]])
    epsilon_schedule = np.reshape(epsilon_schedule, (len(epsilon_schedule), 2))

    # smoothed value
    cumulative_reward = df['cumulative_reward'].values
    min_dst = df['min_dst'].values
    is_crashed = df['is_crashed'].values*1
    is_success = df['is_success'].values*1

    #smooth_reward = savgol_filter(cumulative_reward, window_length=window_size, polyorder=3)
    smooth_reward = running_avg(cumulative_reward,window_size)
    #smooth_min_dst = savgol_filter(min_dst, window_length=window_size, polyorder=3)
    smooth_min_dst = running_avg(min_dst,window_size)
    #smooth_is_crashed = savgol_filter(is_crashed, window_length=window_size, polyorder=3)
    smooth_is_crashed_small = running_avg(is_crashed,small_window)
    smooth_is_crashed_medium = running_avg(is_crashed, medium_window)
    smooth_is_crashed_large = running_avg(is_crashed, large_window)
    smooth_is_crashed_huge = running_avg(is_crashed, huge_window)
    #smooth_is_success = savgol_filter(is_success, window_length=window_size, polyorder=3)
    smooth_is_success_small = running_avg(is_success,small_window)
    smooth_is_success_medium = running_avg(is_success, medium_window)
    smooth_is_success_large = running_avg(is_success, large_window)
    smooth_is_success_huge = running_avg(is_success, huge_window)
    # standard deviation over a moving window for values
    std_poly_reward = create_std_polygon_verts(smooth_reward, cumulative_reward, window_size)
    std_poly_min_dst = create_std_polygon_verts(smooth_min_dst, min_dst, window_size)
    #std_poly_is_crashed = create_std_polygon_verts(smooth_is_crashed, min_dst, window_size)
    #std_poly_is_success = create_std_polygon_verts(smooth_is_success, min_dst, window_size)


    sns.set_theme()
    fig = plt.figure(0, figsize=(14,8))
    ax1 = fig.add_subplot(111)
    ax1.plot(df['ep_num'].values, smooth_reward, color='tab:blue')
    ax1.plot(df['ep_num'].values, cumulative_reward, '--', alpha=0.3, color='tab:blue')

    # graph std tolerance
    polygon = Polygon(std_poly_reward, True, label='Std')
    bp = [polygon]

    p = PatchCollection(bp, alpha=0.2)
    p.set_color('tab:blue')
    ax1.add_collection(p)

    ax1.set_xlabel('Episode Number [-]')
    ax1.set_ylabel('Episode Cumulative Reward [-]')

    ax2 = ax1.twinx()
    ax2.plot(epsilon_schedule[:, 0], epsilon_schedule[:, 1], color='tab:olive')
    ax2.set_ylabel('Epsilon Schedule', color='tab:olive')
    ax2.set_ylim([0,1])

    plt.savefig('..\\Output\\' + str(trial_group) + '\\' + str(trial_number) + '\\Progress\\Graphs\\CumulativeReward.png')

    # ------------------------------------------------------------------------------------------------------------------
    # minimum distance to goal -----------------------------------------------------------------------------------------
    fig = plt.figure(1, figsize=(14, 8))
    ax1 = fig.add_subplot(111)

    ax1.plot(df['ep_num'].values, smooth_min_dst, color='tab:blue')
    ax1.plot(df['ep_num'].values, min_dst, '--', alpha=0.3, color='tab:blue')
    ax1.plot(df['ep_num'].values,np.ones_like(df['ep_num'].values)*5.0,'--',color='tab:orange',label='Goal Distance')

    # graph std tolerance
    polygon = Polygon(std_poly_min_dst, True, label='Std')
    bp = [polygon]

    p = PatchCollection(bp, alpha=0.2)
    p.set_color('tab:blue')
    ax1.add_collection(p)

    ax1.legend()
    ax1.set_xlabel('Episode Number [-]')
    ax1.set_ylabel('Minimum Distance to Destination [m]')

    ax2 = ax1.twinx()
    ax2.plot(epsilon_schedule[:, 0], epsilon_schedule[:, 1], color='tab:olive')
    ax2.set_ylabel('Epsilon Schedule', color='tab:olive')
    ax2.set_ylim([0, 1])

    plt.savefig('..\\Output\\' + str(trial_group) + '\\' + str(trial_number) + '\\Progress\\Graphs\\MinimumDistanceToDestination.png')

    # ------------------------------------------------------------------------------------------------------------------
    # crash rate -----------------------------------------------------------------------------------------
    fig = plt.figure(2, figsize=(14, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(df['ep_num'].values, smooth_is_crashed_small, color='tab:blue',label=str(small_window))
    ax1.plot(df['ep_num'].values, smooth_is_crashed_medium, color='k', label=str(medium_window))
    ax1.plot(df['ep_num'].values, smooth_is_crashed_large, color='tab:purple', label=str(large_window))
    ax1.plot(df['ep_num'].values, smooth_is_crashed_huge, color='tab:green', label=str(huge_window))
    ax1.plot(df['ep_num'].values, is_crashed, '--', alpha=0.2, color='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(epsilon_schedule[:, 0], epsilon_schedule[:, 1], color='tab:olive')
    ax2.set_ylabel('Epsilon Schedule', color='tab:olive')
    y_lims = ax1.get_ylim()
    ax2.set_ylim(y_lims)

    ax1.legend()
    ax1.set_title('Crash Rate with Window Size')
    ax1.set_xlabel('Episode Number [-]')
    ax1.set_ylabel('Crash Rate [-]')
    plt.savefig('..\\Output\\' + str(trial_group) + '\\' + str(trial_number) + '\\Progress\\Graphs\\CrashRate.png')

    # ------------------------------------------------------------------------------------------------------------------
    # success -----------------------------------------------------------------------------------------
    fig = plt.figure(3, figsize=(14, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(df['ep_num'].values, smooth_is_success_small, color='tab:blue',label=str(small_window))
    ax1.plot(df['ep_num'].values, smooth_is_success_medium, color='k', label=str(medium_window))
    ax1.plot(df['ep_num'].values, smooth_is_success_large, color='tab:purple', label=str(large_window))
    ax1.plot(df['ep_num'].values, smooth_is_success_huge, color='tab:green', label=str(huge_window))
    ax1.plot(df['ep_num'].values, is_success, '--', alpha=0.2, color='tab:blue')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(epsilon_schedule[:,0],epsilon_schedule[:,1],color='tab:olive')
    ax2.set_ylabel('Epsilon Schedule',color='tab:olive')
    y_lims = ax1.get_ylim()
    ax2.set_ylim(y_lims)


    ax1.set_title('Success Rate with Window Size')
    ax1.set_xlabel('Episode Number [-]')
    ax1.set_ylabel('Success Rate [-]')

    #plt.show()
    plt.savefig('..\\Output\\'+str(trial_group)+'\\'+str(trial_number)+'\\Progress\\Graphs\\SuccessRate.png')