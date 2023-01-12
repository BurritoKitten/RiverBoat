"""
This script creates a static graph of the episode with more telemetry than the rendered video does. It is also much
faster to visualize the results
"""

# native modules

# 3rd party modules
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import seaborn as sns

# own modules

def graph_episode(k, df, trial_group, trial_number, file_name):


    sns.set_theme()
    fig = plt.figure(0, figsize=(14, 10))
    spec = gridspec.GridSpec(ncols=4, nrows=4, figure=fig)
    ax1 = fig.add_subplot(spec[0:2, 0:2])
    ax2 = fig.add_subplot(spec[0, 2])
    ax3 = fig.add_subplot(spec[0, 3])
    ax4 = fig.add_subplot(spec[1, 2])
    ax5 = fig.add_subplot(spec[1, 3])

    ax6 = fig.add_subplot(spec[2, 0])
    ax7 = fig.add_subplot(spec[2, 1])
    ax8 = fig.add_subplot(spec[2, 2])
    ax9 = fig.add_subplot(spec[2, 3])

    ax10 = fig.add_subplot(spec[3, 0])
    ax11 = fig.add_subplot(spec[3, 1])
    ax12 = fig.add_subplot(spec[3, 2])
    ax13 = fig.add_subplot(spec[3, 3])


    # graph trajectory
    circle = patches.Circle((df['destination_x'].iloc[0], df['destination_y'].iloc[0]), radius=2.0, alpha=1.0,
                            color='tab:green')
    ax1.add_patch(circle)

    sc = ax1.scatter(df['x_pos'], df['y_pos'], c=df['time'], cmap=cm.plasma, edgecolor='none')
    plt.colorbar(sc, ax=ax1)
    # get the points for each 20% of the trajectory
    spacing = int(len(df) / 4)
    idx = [0, spacing, 2 * spacing, 3 * spacing, len(df) - 1]
    for j in range(len(idx)):
        ax1.text(df['x_pos'].iloc[idx[j]], df['y_pos'].iloc[idx[j]], '{:0.1f}'.format(df['time'].iloc[idx[j]]), c='black')

    # ax1.legend()
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')



    plt.suptitle('Trajectories for Episode=' + str(episode) + ' Trial Group=' + str(
        trial_group) + ' Trial Number=' + str(trial_number))
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # Edit this block to control what is rendered ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # input controls.
    trial_group = 'Debugging'
    trial_number = 0
    episodes = [1998]  # range(20)

    # ------------------------------------------------------------------------------------------------------------------
    # Edit this block to control what is rendered ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    for k, episode in enumerate(episodes):
        print('Graphing Episode >> '+str(episode))
        history = pd.read_csv(
            '..\\Output\\' + str(trial_group) + '\\' + str(trial_number) + '\\TrainingHistory\\Data\\History_' + str(
                episode) + '.csv')

        ge = graph_episode(k, df=history, trial_group=trial_group, trial_number=trial_number,
                            file_name='..\\Output\\' + str(trial_group) + '\\' + str(
                                trial_number) + '\\TrainingHistory\\Graphs\\History_' + str(episode) + '.png')


