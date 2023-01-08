"""
This script renders a video and creates a graph for a single episode or a combination of episodes that have already
been run. The primary use is for visualization, demonstration, and debugging of the simulations.
This script does not render baseline, robust baselines, or overall training metrics.
"""

# native modules
import copy

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

    def __init__(self, df, trial_group, trial_number, file_name):
        """

        :param df:
        """

        # data of the episode
        self.df = df

        # output file location and name
        self.file_name = file_name

        #tolerance = generate_buffer_verts(nav_points, buffer=5.0)

        # get the maximum and minimum values of the simulation
        max_x = 150.0 #max([self.df['x_pos'].max,self.df['static_circle_0_x'].max])
        min_x = -50.0 #max([self.df['x_pos'].min, self.df['static_circle_0_x'].min])
        max_y = 150.0
        min_y = -50.0

        # loop helper
        self.c = 0

        sns.set_theme()
        fig = plt.figure(0, figsize=(14, 10))
        spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
        ax1 = fig.add_subplot(spec[0:2, 0:2])
        ax2 = fig.add_subplot(spec[2, 0])
        ax3 = fig.add_subplot(spec[2, 1])

        steps = len(df) - 1
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Nathan'), bitrate=1800)

        def animate(steps):
            print('Frame >> ' + str(self.c))
            self.c = self.c + 1

            ax1.clear()
            ax2.clear()
            ax3.clear()


            # plot the boats current position
            # boats x,y location [m]
            bx = self.df['x_pos'].iloc[self.c-1]
            by = self.df['y_pos'].iloc[self.c-1]
            hull_len = self.df['hull_length'].iloc[self.c-1]
            hull_width = self.df['hull_width'].iloc[self.c - 1]
            psi = self.df['psi'].iloc[self.c-1]
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

            p = PatchCollection(bp, alpha=0.2)
            p.set_color('tab:blue')
            ax1.add_collection(p)

            # draw boats trajectory
            x_traj = self.df['x_pos'].values
            y_traj = self.df['y_pos'].values
            ax1.plot(x_traj[:self.c-1],y_traj[:self.c-1],color='tab:purple',label='Trajectory')

            # draw propeller angle
            delta = self.df['delta'].iloc[self.c-1]
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

            # draw path if applicable
            try:
                # draw the path
                cp = self.df['action_meta_data'].iloc[self.c-1]
                cp = cp.split(';')

                for i, tmp_cp in enumerate(cp):
                    if tmp_cp != '':
                        cp[i] = tmp_cp.split('_')
                        cp[i] = [float(item) for item in cp[i]]
                    else:
                        cp.pop(i)

                cp = np.reshape(cp,(len(cp),2))
                path = bezier_curve(cp, 20)
                tolerance = generate_buffer_verts(path,buffer=5.0)
                path_tolerance = Polygon(tolerance, True)
                patches = []
                patches.append(path_tolerance)

                # path control points
                ax1.scatter(cp[:,0],cp[:,1],color='tab:olive')

                # mean path
                ax1.plot(path[:,0],path[:,1],'--',color='tab:olive',label='Path')

                p = PatchCollection(patches, alpha=0.2)
                p.set_color('tab:olive')
                ax1.add_collection(p)
            except:
                pass


            # draw the destination
            circle_destination = plt.Circle((self.df['destination_x'][self.c-1], self.df['destination_y'][self.c-1]), 5.0, color='tab:green',alpha=0.2,label='Destination')
            ax1.add_patch(circle_destination)

            # draw the obstacles
            obstacles = [name for name in self.df.columns if 'static_circle' in name]
            unique_obs = []
            for obs in obstacles:
                parts = obs.split('_')
                if int(parts[2]) not in unique_obs:
                    unique_obs.append(int(parts[2]))
            for part in unique_obs:
                circle_destination = plt.Circle(
                    (self.df['static_circle_'+str(part)+'_x'][self.c - 1], self.df['static_circle_'+str(part)+'_y'][self.c - 1]), 10.0,
                    color='tab:red', alpha=0.2, label='Obstacle')
                ax1.add_patch(circle_destination)

            #ax1.add_collection(p)
            #boat_patch = mpatches.Patch(color='tab:blue', alpha=0.2, label='Boat')
            #ax1.legend(handles=[boat_patch])
            ax1.legend()
            ax1.set_xlim([min_x,max_x])
            ax1.set_ylim([min_y, max_y])

            # plot the reward
            time = self.df['time'].values
            reward = self.df['reward'].values
            ax2.plot(time[:self.c - 1], reward[:self.c - 1], color='tab:blue', label='Trajectory')
            ax2.set_ylabel('Reward [-]')
            ax2.set_xlabel('Time [s]')
            ax2.set_xlim([min(time),max(time)])
            if min(reward) != max(reward):
                ax2.set_ylim([min(reward)-0.05*(max(reward)-min(reward)), max(reward)+0.05*(max(reward)-min(reward))])


            # plot distance to the goal
            dest_dist = self.df['dest_dist'].values
            ax3.plot(time[:self.c - 1], dest_dist[:self.c - 1], color='tab:blue', label='Trajectory')
            ax3.set_ylabel('Dist to Dest [m]')
            ax3.set_xlabel('Time [s]')
            ax3.set_xlim([min(time), max(time)])
            ax3.set_ylim([min(dest_dist)-0.05*(max(dest_dist)-min(dest_dist)), max(dest_dist)+0.05*(max(dest_dist)-min(dest_dist))])

            # plot distance to any obstacles

            plt.suptitle('Trajectories for Episode=' + str(episode) + ' Trial Group=' + str(
                trial_group) + ' Trial Number=' + str(trial_number) )

        # create the video
        ani = animation.FuncAnimation(fig, animate, frames=steps, interval=20, blit=False)

        # save the video
        ani.save(self.file_name, writer=writer)

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # Edit this block to control what is rendered ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # input controls.
    trial_group = 'Debugging'
    trial_number = 0
    episode = 16

    # ------------------------------------------------------------------------------------------------------------------
    # Edit this block to control what is rendered ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    history = pd.read_csv('..\\Output\\'+str(trial_group)+'\\'+str(trial_number)+'\\TrainingHistory\\Data\\History_'+str(episode)+'.csv')

    ae = AnimateEpisode(df=history, trial_group=trial_group, trial_number=trial_number, file_name='..\\Output\\'+str(trial_group)+'\\'+str(trial_number)+'\\TrainingHistory\\Videos\\History_'+str(episode)+'.mp4')

