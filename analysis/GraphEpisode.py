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
    ax2 = fig.add_subplot(spec[0:2, 2:4])

    ax3 = fig.add_subplot(spec[2:4, 0:2])
    ax4 = fig.add_subplot(spec[2, 2])
    ax5 = fig.add_subplot(spec[2, 3])

    ax6 = fig.add_subplot(spec[3, 2])
    ax7 = fig.add_subplot(spec[3, 3])


    # graph trajectory
    circle = patches.Circle((df['destination_x'].iloc[0], df['destination_y'].iloc[0]), radius=5.0, alpha=0.3,
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

    # graph q values
    q_val_header = [n for n in list(df.columns) if 'Network_output' in n]
    df_q_vals = df[q_val_header]
    df_q_vals['mean'] = df_q_vals.mean(axis=1)
    df_q_vals['min'] = df_q_vals.min(axis=1)
    df_q_vals['max'] = df_q_vals.max(axis=1)
    ax2.plot(df['time'].values,df_q_vals['max'],label='max')
    ax2.plot(df['time'].values, df_q_vals['min'], label='min')
    ax2.plot(df['time'].values, df_q_vals['mean'], label='mean')
    ax2.legend()
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Q value')

    # graph angles
    ax3.plot(df['time'].values,np.rad2deg(df['psi'].values),label='$\phi$ hull angle')
    ax3.plot(df['time'].values, np.rad2deg(df['delta'].values), label='$\delta$ propeller angle')
    ax3.plot(df['time'].values, np.rad2deg(df['mu'].values), label='$\mu$ angle to dest')
    ax3.plot([0,max(df['time'].values)],[45,45],'k--',label='Max Propeller')
    ax3.plot([0, max(df['time'].values)], [-45, -45], 'k--')
    ax3.legend()
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Angle [deg]')

    # reward
    ax4.plot(df['time'].values, df['reward'].values)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Reward [-]')

    # velocities
    ax5.plot(df['time'].values, df['v_xp'].values,label='x_p')
    ax5.plot(df['time'].values, df['v_yp'].values,label='y_p')
    ax5.legend()
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Velocities [m/s]')

    # velocities
    ax6.plot(df['time'].values, df['dest_dist'].values)
    ax6.plot([0, max(df['time'].values)], [5.0,5.0], label='Threshold')
    ax6.legend()
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Distance to Destination [m]')

    # fuel
    ax7.plot(df['time'].values, df['fuel'].values)
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Fuel [kg]')

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
    trial_group = 'DebuggingPath'
    trial_number = 1
    episodes = range(9900,10000)

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


