"""
create graphs of the boat for what defines the possible are a boat could be in while traveling for 10 seconds
"""

# native modules

# 3rd party code
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import seaborn as sns

# own code
import src.Movers as movers

if __name__ == '__main__':

    # create the boat
    delta_t = 0.1 # simulation time step
    rb = movers.RiverBoat.get_default(delta_t)

    # define the initial conditions
    initial_conditions = dict()
    initial_conditions['x_pos'] = 0.0
    initial_conditions['y_pos'] = 0.0
    initial_conditions['v_xp'] = 0.25
    initial_conditions['v_yp'] = 0.0
    initial_conditions['v_x'] = 0.0
    initial_conditions['v_y'] = 0.0
    initial_conditions['psi'] = np.deg2rad(45.0)

    # reset all of the information of the boat
    rb.initalize_in_state_dict()

    # set initial conditions to desited state
    for key, item in initial_conditions.items():
        rb.state_dict[key] = item

    # get the maximum distance a boat can travel
    max_t = 10.0
    t = 0
    while t < max_t:
        rb.step(t)

        t += delta_t
    max_dst = np.sqrt(rb.state_dict['x_pos']**2 + rb.state_dict['y_pos']**2)

    # directly stragith maximum location
    max_x = rb.state_dict['x_pos']
    max_y = rb.state_dict['y_pos']
    far_angle = np.arctan2(max_y,max_x)


    # range of propeller angles to sweep over for polygon
    delta_values = np.deg2rad(np.linspace(-45,45,51))
    # data to extract from simulations
    x_end = []
    x_paths = []
    y_end = []
    y_paths = []
    for delta in delta_values:

        # reset all of the information of the boat
        rb.initalize_in_state_dict()

        # set initial conditions to desited state
        for key, item in initial_conditions.items():
            rb.state_dict[key] = item

        # set initial propeller angle
        rb.state_dict['delta'] = delta

        # run simulation for 10 seconds
        t = 0.0
        tmp_x_path = [rb.state_dict['x_pos']]
        tmp_y_path = [rb.state_dict['y_pos']]
        while t < max_t:

            rb.step(t)

            tmp_x_path.append(rb.state_dict['x_pos'])
            tmp_y_path.append(rb.state_dict['y_pos'])

            t += delta_t

        # get ending location of the boat
        x_end.append(rb.state_dict['x_pos'])
        y_end.append(rb.state_dict['y_pos'])
        x_paths.append(tmp_x_path)
        y_paths.append(tmp_y_path)

    sns.set_theme()
    fig = plt.figure(1,figsize=(14,8))
    ax = fig.add_subplot(111)

    end_points = np.zeros((len(x_end),2))
    end_points[:,0] = x_end
    end_points[:,1] = y_end
    #end_points = np.vstack([end_points, [initial_conditions['x_pos'],initial_conditions['y_pos']]])
    path_tolerance = Polygon(end_points, True)
    bp = [path_tolerance]

    p = PatchCollection(bp, alpha=0.4)
    p.set_color('tab:blue')
    ax.add_collection(p)

    end_points = np.vstack([end_points, [initial_conditions['x_pos'], initial_conditions['y_pos']]])
    path_tolerance = Polygon(end_points, True)
    bp = [path_tolerance]

    p = PatchCollection(bp, alpha=0.1)
    p.set_color('tab:red')
    ax.add_collection(p)

    ax.plot([0], [0], 'o', label='Start')
    #ax.scatter(x_end,y_end,label='Ending Location')

    #for i, path in enumerate(x_paths):
    #    ax.plot(x_paths[i],y_paths[i],label=str(np.rad2deg(delta_values[i])))

    # graph maximum circle
    x_max = []
    y_max = []
    theta = np.linspace(0,2.0*np.pi,360)
    for the in theta:
        x_max.append(max_dst*np.cos(the))
        y_max.append(max_dst * np.sin(the))
    ax.plot(x_max,y_max,'--',label='Max Distance')

    # generate a sector of possible future locations
    max_angle_off = 0.0
    for i, _ in enumerate(x_end):
        tmp_theta = np.arctan2(y_end[i], x_end[i])
        tmp_angle_off =  np.abs(far_angle-tmp_theta)

        if tmp_angle_off > max_angle_off:
            max_angle_off = tmp_angle_off

    diff_low = far_angle-max_angle_off
    diff_high = far_angle + max_angle_off
    plt.plot([0,max_dst*np.cos(diff_low)],[0,max_dst*np.sin(diff_low)],':')
    plt.plot([0, max_dst * np.cos(diff_high)], [0, max_dst * np.sin(diff_high)], ':')

    plt.legend()
    plt.show()