"""
This script renders a video and creates a graph for a single episode or a combination of episodes that have already
been run. The primary use is for visualization, demonstration, and debugging of the simulations.
This script does not render baseline, robust baselines, or overall training metrics.
"""

# native modules
import copy
import os

# 3d party modules
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# own modules
from src.ActionOperation import bezier_curve


def generate_buffer_verts(samples, buffer):
    """
    takes a list of points that describe a path and creates a zone around the path
    :param samples: the list of (x,y) points
    :param buffer: the size in meters of the bandwidth allowed around the boat
    :return:
    """
    upper = []
    lower = []
    for i, td in enumerate(samples):

        if i == len(samples) - 1:
            tmp_vec = [samples[i, 0] - samples[i - 1, 0], samples[i, 1] - samples[i - 1, 1]]
        elif i == 0:
            tmp_vec = [samples[i + 1, 0] - samples[i, 0], samples[i + 1, 1] - samples[i, 1]]
        else:
            tmp_vec = [samples[i + 1, 0] - samples[i - 1, 0], samples[i + 1, 1] - samples[i - 1, 1]]

        tmp_vec = tmp_vec / np.linalg.norm(tmp_vec) * buffer
        tmp_vec = np.array(tmp_vec)

        neg_angle = -np.pi / 2.0
        rot_mat = [[np.cos(neg_angle), -np.sin(neg_angle)],
                   [np.sin(neg_angle), np.cos(neg_angle)]]
        rot_mat = np.reshape(rot_mat, (2, 2))
        offset = np.matmul(rot_mat, tmp_vec)

        lower.append([samples[i, 0] + offset[0], samples[i, 1] + offset[1]])

        angle = np.pi / 2.0
        rot_mat = [[np.cos(angle), -np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]]
        rot_mat = np.reshape(rot_mat, (2, 2))
        offset = np.matmul(rot_mat, tmp_vec)

        upper.append([samples[i, 0] + offset[0], samples[i, 1] + offset[1]])

    lower = np.reshape(lower, (len(lower), 2))
    upper = np.reshape(upper, (len(upper), 2))

    upper = np.flip(upper, axis=0)
    verts = np.concatenate((lower, upper))

    return verts

class AnimateEpisode:

    def __init__(self, ep_count, df_lst, trial_group, trial_number, file_name):
        """

        :param df:
        """

        # data of the episode
        self.df_lst = df_lst

        # output file location and name
        self.file_name = file_name

        #tolerance = generate_buffer_verts(nav_points, buffer=5.0)

        # get the maximum and minimum values of the simulation
        max_x = 300.0
        min_x = -50.0
        max_y = 250.0
        min_y = -50.0

        # loop helper
        self.c = 0
        self.set_size = len(df_lst)

        sns.set_theme()
        fig = plt.figure(k, figsize=(14, 10))
        spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
        ax1 = fig.add_subplot(spec[0:2, 0:2])
        #ax2 = fig.add_subplot(spec[2, 0])
        #ax3 = fig.add_subplot(spec[2, 1])

        steps = 0
        for df in df_lst:
            tmp_steps = len(df)-1
            if tmp_steps > steps:
                steps = tmp_steps
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Nathan'), bitrate=1800)

        cmap = cm.get_cmap('plasma')

        def animate(steps):
            print('Ep=' + str(ep_count) + ' Frame >> ' + str(self.c))

            ax1.clear()
            #ax2.clear()
            #ax3.clear()

            for k, df in enumerate(self.df_lst):

                if self.c >= len(df):
                    max_idx = len(df)-1
                else:
                    max_idx = self.c

                # draw path if applicable
                '''
                try:
                    # draw the path
                    cp = df['path'].iloc[max_idx]
                    cp = cp.split(';')

                    for i, tmp_cp in enumerate(cp):
                        if tmp_cp != '':
                            cp[i] = tmp_cp.split('_')
                            cp[i] = [float(item) for item in cp[i]]
                        else:
                            cp.pop(i)

                    cp = np.reshape(cp, (len(cp), 2))
                    path = bezier_curve(cp, 20)
                    tolerance = generate_buffer_verts(path, buffer=5.0)
                    path_tolerance = Polygon(tolerance, True)
                    patches = []
                    patches.append(path_tolerance)

                    # path control points
                    ax1.scatter(cp[:, 0], cp[:, 1], color='tab:olive')

                    # mean path
                    ax1.plot(path[:, 0], path[:, 1], '--', color='tab:olive', label='Path')

                    p = PatchCollection(patches, alpha=0.2)
                    p.set_color('tab:olive')
                    ax1.add_collection(p)
                except:
                    pass
                '''



                # plot the boats current position
                # boats x,y location [m]

                #print(len(df),max_idx)
                bx = df['x_pos'].iloc[max_idx]
                by = df['y_pos'].iloc[max_idx]
                hull_len = df['hull_length'].iloc[max_idx]
                hull_width = df['hull_width'].iloc[max_idx]
                psi = df['psi'].iloc[max_idx]
                corners = [[hull_len / 2.0, -hull_width / 2.0],
                           [hull_len / 2.0, hull_width / 2.0],
                           [-hull_len / 2.0, hull_width / 2.0],
                           [-hull_len / 2.0, -hull_width / 2.0]]

                for i, corn in enumerate(corners):
                    corn_new = copy.deepcopy(corn)
                    corn[0] = corn_new[0] * np.cos(psi) - corn_new[1] * np.sin(psi) + bx
                    corn[1] = corn_new[0] * np.sin(psi) + corn_new[1] * np.cos(psi) + by

                corners = np.reshape(corners, (len(corners), 2))
                polygon = Polygon(corners, True, label='Obstacle')
                bp = [polygon]

                p = PatchCollection(bp, alpha=0.4)
                p.set_color(cmap(k/self.set_size))
                ax1.add_collection(p)

                # draw boats trajectory
                x_traj = df['x_pos'].values
                y_traj = df['y_pos'].values
                ax1.plot(x_traj[:max_idx],y_traj[:max_idx],color=cmap(k/self.set_size),label='Trajectory')

                # draw propeller angle
                delta = df['delta'].iloc[max_idx]
                prop_end_x = -hull_len / 2.0
                prop_end_y = 0.0
                prop_end_x_new = prop_end_x * np.cos(psi) - prop_end_y * np.sin(psi) + bx
                prop_end_y_new = prop_end_x * np.sin(psi) + prop_end_y * np.cos(psi) + by
                base_vec = [prop_end_x_new - bx, prop_end_y_new - by]
                base_vec = base_vec / np.linalg.norm(base_vec) * 2.0
                rot_mat = [[np.cos(delta), -np.sin(delta)],
                           [np.sin(delta), np.cos(delta)]]
                rot_mat = np.reshape(rot_mat, (2, 2))
                base_vec = np.matmul(rot_mat, base_vec)

                ax1.plot([prop_end_x_new, prop_end_x_new + base_vec[0]], [prop_end_y_new, prop_end_y_new + base_vec[1]],
                        color='tab:blue')

                # draw the destination
                circle_destination = plt.Circle((df['destination_x'][max_idx], df['destination_y'][max_idx]), 5.0, color='tab:green',alpha=0.2,label='Destination')
                ax1.add_patch(circle_destination)

                # draw the obstacles
                obstacles = [name for name in df.columns if 'static_circle' in name]
                unique_obs = []
                for obs in obstacles:
                    parts = obs.split('_')
                    if int(parts[2]) not in unique_obs:
                        unique_obs.append(int(parts[2]))
                for part in unique_obs:
                    circle_destination = plt.Circle(
                        (df['static_circle_'+str(part)+'_x'][max_idx], df['static_circle_'+str(part)+'_y'][max_idx]), 10.0,
                        color='tab:red', alpha=0.2, label='Obstacle')
                    ax1.add_patch(circle_destination)

                #ax1.legend()
                ax1.set_xlim([min_x,max_x])
                ax1.set_ylim([min_y, max_y])

                # plot the reward
                '''
                time = df['time'].values
                reward = df['reward'].values
                ax2.plot(time, reward, color=cmap(k/self.set_size), label='Trajectory')
                ax2.scatter(time[max_idx], reward[max_idx], color='k')
                ax2.set_ylabel('Reward [-]')
                ax2.set_xlabel('Time [s]')
                ax2.set_xlim([min(time),max(time)+1])
                if min(reward) != max(reward):
                    ax2.set_ylim([min(reward)-0.05*(max(reward)-min(reward)), max(reward)+0.05*(max(reward)-min(reward))])


                # plot distance to the goal
                dest_dist = df['dest_dist'].values
                #ax3.plot(time[:self.c], dest_dist[:self.c], color='tab:blue', label='Trajectory')
                ax3.plot(time, dest_dist, color=cmap(k/self.set_size), label='Trajectory')
                ax3.scatter(time[max_idx], dest_dist[max_idx],color='k')
                ax3.set_ylabel('Dist to Dest [m]')
                ax3.set_xlabel('Time [s]')
                ax3.set_xlim([min(time), max(time)+1])
                ax3.set_ylim([min(dest_dist)-0.05*(max(dest_dist)-min(dest_dist)), max(dest_dist)+0.05*(max(dest_dist)-min(dest_dist))])
                '''

                # plot distance to any obstacles

                plt.suptitle('Trajectories for Episode=' + str(ep_count) + ' Trial Group=' + str(
                    trial_group) + ' Trial Number=' + str(trial_number) )

            self.c = self.c + 1

            # close dataframe loop

        # create the video
        ani = animation.FuncAnimation(fig, animate, frames=steps, interval=0.1, blit=False)

        # save the video
        ani.save(self.file_name, writer=writer)

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # Edit this block to control what is rendered ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # input controls.
    trial_group = 'DebuggingPathDDPG'
    trial_number = 11
    episodes = [1500, 2000]  # min and max values

    # ------------------------------------------------------------------------------------------------------------------
    # Edit this block to control what is rendered ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # get the episodes that had evaluation simulations
    avg_progress = pd.read_csv(
        '..\\Output\\' + str(trial_group) + '\\' + str(trial_number) + '\\Progress\\Data\\evaluation_average.csv')
    ep_list = list(avg_progress['EpNum'].values)
    ep_list = [i for i in ep_list if i >= episodes[0] and i <= episodes[1] ]

    # get the file names in the evaluation folder
    file_lst = os.listdir('..\\Output\\' + str(trial_group) + '\\' + str(trial_number) + '\\Evaluation\\Data')
    num_evals_per_step = len([i for i in file_lst if 'History_0-' in i])

    k = 0
    for k in range(len(ep_list)):
        history_lst = []
        for i in range(num_evals_per_step):
            # get a list of histories
            df = pd.read_csv('..\\Output\\' + str(trial_group) + '\\' + str(trial_number) + '\\Evaluation\\Data\\History_'+str(ep_list[k])+'-'+str(i)+'.csv')
            history_lst.append(df)

        ae = AnimateEpisode(k, df_lst=history_lst, trial_group=trial_group, trial_number=trial_number,
                            file_name='..\\Output\\' + str(trial_group) + '\\' + str(
                                trial_number) + '\\Evaluation\\Videos\\History_' + str(ep_list[k]) + '.mp4')


