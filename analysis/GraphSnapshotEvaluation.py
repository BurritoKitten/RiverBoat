"""
Used to create a graph for describing the evaluation set in the paper
"""

import copy
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import os
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    trial_group = 'TuneDDPGPathControlSparseReward'
    trial_number = 10
    episodes = [7500, 7501]  # min and max values

    # get the episodes that had evaluation simulations
    avg_progress = pd.read_csv(
        '..\\Output\\' + str(trial_group) + '\\' + str(trial_number) + '\\Progress\\Data\\evaluation_average.csv')
    ep_list = list(avg_progress['EpNum'].values)
    ep_list = [i for i in ep_list if i >= episodes[0] and i <= episodes[1]]

    # get the file names in the evaluation folder
    file_lst = os.listdir('..\\Output\\' + str(trial_group) + '\\' + str(trial_number) + '\\Evaluation\\Data')
    num_evals_per_step = len([i for i in file_lst if 'History_0-' in i])

    k = 0
    for k in range(len(ep_list)):
        history_lst = []
        for i in range(num_evals_per_step):
            # get a list of histories
            df = pd.read_csv(
                '..\\Output\\' + str(trial_group) + '\\' + str(trial_number) + '\\Evaluation\\Data\\History_' + str(
                    ep_list[k]) + '-' + str(i) + '.csv')
            history_lst.append(df)
        sns.set_theme()
        file_name='C:/Users/nhemm/Documents/PhD/Papers/Paper2/08_Images/EvaluationDescription.png'
        fig = plt.figure(0)
        ax1 = fig.add_subplot(111)
        time_to_graph = 100
        cmap = cm.get_cmap('plasma')
        set_size = len(history_lst)
        for k, df in enumerate(history_lst):

            # draw path if applicable
            if time_to_graph > len(df):
                max_idx = len(df)-1
            else:
                max_idx = time_to_graph

            # plot the boats current position
            # boats x,y location [m]

            # print(len(df),max_idx)
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
            p.set_color(cmap(k / set_size))
            ax1.add_collection(p)

            # draw boats trajectory
            x_traj = df['x_pos'].values
            y_traj = df['y_pos'].values
            ax1.plot(x_traj[:max_idx], y_traj[:max_idx], color=cmap(k / set_size), label='Trajectory')

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
            if k == 0:
                circle_destination = plt.Circle((df['destination_x'][max_idx], df['destination_y'][max_idx]), 5.0,
                                                color='tab:green', alpha=0.2, label='Destination')
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
                        (df['static_circle_' + str(part) + '_x'][max_idx],
                         df['static_circle_' + str(part) + '_y'][max_idx]), 20.0,
                        color='tab:gray', alpha=0.2, label='Obstacle')
                    ax1.add_patch(circle_destination)

        # ax1.legend()
        ax1.set_xlim([0, 200])
        ax1.set_ylim([50, 150])
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')


        plt.suptitle('Snap Shot Showing Initial Conditions Of Evaluation Set')
        plt.savefig(file_name)
        plt.show()