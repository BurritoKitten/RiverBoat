"""
Movers holds the objects that move and make make decisions when moving in an environment. Primarily, this is targeted
at the river boat and obstacles that are not river boats.

"""

from abc import ABC, abstractmethod
from collections import namedtuple, OrderedDict
import os

import numpy as np
import pandas as pd

class Mover(ABC):

    def __init__(self,name):

        self.name = name
        self.state_dict = dict()
        self.sensors = []
        self.history = pd.DataFrame()
        self.can_learn = False

    @abstractmethod
    def step(self):
        """
        steps the state of the mover object forward by one time step.
        :return:
        """
        raise NotImplementedError('The mover '+self.name+' needs to implement the step method')

    def set_domain(self, domain):
        """
        sets the possible domain the mover can exists in the simulation
        :param domain:
        :return:
        """

        # domain has information of:
        #     | x_min, x_max |
        #     | y_min, y_max |
        if domain.shape() != (2, 2):
            raise ValueError('The domain must have a shape of (2,2)')
        self.domain = domain

    def add_sensor(self,sensor):
        """
        add a sensor object to the mover. All sensors are stored in a list and are updated in order that they are added
        to the mover
        :param sensor: the sensor object that samples the enviornment to update its measurements
        :return:
        """
        self.sensors.append(sensor)

    def remove_sensor(self,sensor_name):
        """
        removes a sensor from the mover's sensors based on the target sensors name
        :param sensor_name:
        :return:
        """
        for i, sensor in enumerate(self.sensors):
            if sensor.name == sensor_name:
                self.sensors.pop(i)
                return True
        return False

    def update_sensors(self, mover_dict):
        """
        loops over each sensor in the mover and updates the measurements for each sensor
        :return:
        """
        for sensor in self.sensors:
            sensor.update_measurements(mover_dict)

    @abstractmethod
    def get_history(self):
        """

        :return:
        """
        return self.history

    @abstractmethod
    def reset_history(self, num_steps):
        raise NotImplementedError('The mover '+self.name+' must implement the reset_history method')

    @abstractmethod
    def add_step_history(self, step_num):
        raise NotImplementedError('The mover '+self.name+' must implement the add_step_history method')

    @abstractmethod
    def normalize_state(self):
        raise NotImplementedError('The mover '+self.name+' must implement the normalize_state method')

    @abstractmethod
    def action_to_command(self):
        """
        takes an action from agent and converts it into the command to control the mover
        :return:
        """
        raise NotImplementedError('The mover ' + self.name + ' must implement the action_to_command method')

class StaticCircleObstacle(Mover,ABC):

    def __init__(self, name, radius, domain):
        """
        creates a Circle shaped obstacle that does not move during the simulation.

        :param name: a string that allows for easy differentiaion and logging of the mover
        :param radius: the radius of the circular obstacle in meters
        :param domain: a tuple of shape (2,2) that denots the min and max, x and y values for the obstacles center to
            be in
        """
        # instantiate mover
        super().__init__(name)

        self.state_dict['x_pos'] = 0.0  # sets the x position of the circle
        self.state_dict['y_pos'] = 0.0  # sets the y position of the circle
        self.state_dict['x_pos_norm'] = 0.0  # initial normalized arbitrary value for the x position of the circle
        self.state_dict['y_pos_norm'] = 0.0  # initial normalized arbitrary value for the y position of the circle
        self.state_dict['radius'] = radius  # radius in meters of the obstacle
        self.set_domain(domain)

    def step(self):
        # no changes are made as this is a static circle
        pass

    def normalize_state(self):
        """
        normalizes the postion and radius of the mover to be between 0 and 1
        :return:
        """
        self.norm_pos[0] = (self.pos[0]-self.domain[0,0])/(self.domain[0,1]-self.domain[0,0])
        self.norm_pos[1] = (self.pos[1] - self.domain[1, 0]) / (self.domain[1, 1] - self.domain[1, 0])

    def action_to_command(self):
        # there are no actions for the static circle as it does not make any decisions
        return 0

    def add_step_history(self, step_num):
        self.history.iloc[step_num] = [self.state_dict['x_pos'], self.state_dict['y_pos']]

    def reset_history(self, num_steps):
        """
        sets all of the history to zero
        :param num_steps:
        :return:
        """
        empty_data = np.zeros((num_steps, 2))
        self.history = pd.DataFrame(data=empty_data, columns=[self.name+'_x',self.name+'_y'])

    def trim_history(self, step_num):
        self.history.drop(range(step_num, len(self.history)), inplace=True)


class RiverBoat(Mover,ABC):

    def __init__(self, name, area_air, area_water, bsfc, delta, delta_max, delta_t, density_air, density_water, fom,
                 fuel, fuel_capacity, hull_len, hull_width, mass, moi, power, power_max, psi, prop_diam):
        self.state_dict = OrderedDict()
        self.initalize_in_state_dict()

        # --------------------------------------------------------------------------------------------------------------
        # set the values of the boat
        # time of the simulation
        self.state_dict['time'] = 0.0
        # name of the boat
        self.state_dict['name'] = name
        # cross sectional area of the part of the boat in the air [m^2]
        self.state_dict['area_air'] = area_air
        # cross sectional area of the part of the boat in the water [m^2]
        self.state_dict['area_water'] = area_water
        # sets the brake specific fuel consumption [kg/w-s]
        self.state_dict['bsfc'] = bsfc
        # propeller angle [rad]
        self.state_dict['delta'] = delta
        # set bounds for the propeller angle
        self.state_dict['delta_max'] = delta_max
        # time step of the simulation
        self.state_dict['delta_t'] = delta_t
        # air density [kg/m^3]
        self.state_dict['density_air'] = density_air
        # water density [kg/m^3]
        self.state_dict['density_water'] = density_water
        # figure of merit of the propeller
        self.state_dict['fom'] = fom
        # sets the current level of fuel on the boat [kg]
        self.state_dict['fuel'] = fuel
        # sets the maximum amount of fuel the boat can have [kg]
        self.state_dict['fuel_capacity'] = fuel_capacity
        # the length of the hull [m]
        self.state_dict['hull_length'] = hull_len
        # the length of the hull [m]
        self.state_dict['hull_width'] = hull_width
        # the total mass of the boat
        self.state_dict['mass'] = mass
        # set the moment of inertia of the boat [kg m^2]
        self.state_dict['moi'] = moi
        # current power level of the propeller [watt]
        self.state_dict['power'] = power
        # maximum power level of the propeller [watt]
        self.state_dict['power_max'] = power_max
        # angle of the hull to the positive x axis in the global frame [rad]
        self.state_dict['psi'] = psi
        # sets the disk area of the propeller counting the area of the spinner in the disk area [m]
        self.state_dict['prop_area'] = np.pi * prop_diam * prop_diam

        # telemetry
        # self.telemetry = pd.DataFrame(self.state_dict.values(),columns=self.state_dict.keys())
        self.telemetry = pd.DataFrame([self.state_dict])

    def initalize_in_state_dict(self):
        # --------------------------------------------------------------------------------------------------------------
        # position data
        # --------------------------------------------------------------------------------------------------------------
        # x position in the global reference frame
        self.state_dict['x_pos'] = 0.0
        # y position in the global reference frame
        self.state_dict['y_pos'] = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # geometric data
        # --------------------------------------------------------------------------------------------------------------
        # relative angle of attack of the propeller disk [deg]
        self.state_dict['alpha'] = 0.0
        # angle [rad] of the propeller relative to the longitudinal axis of the boat
        self.state_dict['delta'] = 0.0
        # angle of the boat hull in the global reference frame where positive x is where the angle is measured too
        self.state_dict['psi'] = 0.0
        # the total mass of the boat [kg]
        self.state_dict['mass'] = 0.0
        # air projected area [m^2]
        self.state_dict['area_air'] = 0.0
        # water projected area [m^2]
        self.state_dict['area_water'] = 0.0
        # moment of interia of the boat []
        self.state_dict['moi'] = 0.0
        # lenght of the hull of the boat
        self.state_dict['hull_length'] = 0.0
        # the widest width of the hull of the boat
        self.state_dict['hull_width'] = 0.0
        # the disk area of the propeller excluding the aread of the spinner
        self.state_dict['prop_area'] = 0.0
        # effective angle of incidence of the air
        self.state_dict['psi_eff_air'] = 0.0
        # effective angle of incidence of the water
        self.state_dict['psi_eff_water'] = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # velocity data
        # --------------------------------------------------------------------------------------------------------------
        # x (longitudinal/surge) velocity [m/s] in the boats local frame
        self.state_dict['v_xp'] = 0.0
        # y (lateral/sway) velocity [m/s] in the boats local frame
        self.state_dict['v_yp'] = 0.0
        # x velocity [m/s] in the global frame
        self.state_dict['v_x'] = 0.0
        # y velocity [m/s] in the global frame
        self.state_dict['v_y'] = 0.0
        # rotational velocity
        self.state_dict['psi_dot'] = 0.0
        # effective longitudinal velocity of the air
        self.state_dict['v_x_eff_air'] = 0.0
        # effective lateral velocity of the air
        self.state_dict['v_y_eff_air'] = 0.0
        # effective longitudinal velocity of the water
        self.state_dict['v_x_eff_water'] = 0.0
        # effective lateral velocity of the water
        self.state_dict['v_y_eff_water'] = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # acceleration data
        # --------------------------------------------------------------------------------------------------------------
        # x (longitudinal/surge) acceleration [m/s^2] in the boats local frame
        self.state_dict['acc_xp'] = 0.0
        # y (lateral/sway) acceleration [m/s^2] in the boats local frame
        self.state_dict['acc_yp'] = 0.0
        # x acceleration [m/s^2] in the global reference frame
        self.state_dict['acc_x'] = 0.0
        # y acceleration [m/s^2] in the global reference frane
        self.state_dict['acc_y'] = 0.0
        # rotational acceleration
        self.state_dict['psi_double_dot'] = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # other data
        # --------------------------------------------------------------------------------------------------------------
        # the power [watt] that is deliverd to the propeller
        self.state_dict['power'] = 0.0
        # the thrust [N] that is produce by the propeller at a given power level
        self.state_dict['thrust'] = 0.0
        # the figure of merit for the propeller
        self.state_dict['fom'] = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # simulation data
        # --------------------------------------------------------------------------------------------------------------
        # time step of the simulation [s]
        self.state_dict['delta_t'] = 0.0
        # the name of this boat
        self.state_dict['name'] = ''
        # wind velocity [m/s]
        self.state_dict['v_wind'] = np.array([0.0, 0.0])
        # current velocity [m/s]
        self.state_dict['v_current'] = np.array([0.0, 0.0])

        # --------------------------------------------------------------------------------------------------------------
        # forces and moments
        # --------------------------------------------------------------------------------------------------------------
        self.state_dict['f_d_air'] = 0.0
        self.state_dict['f_s_air'] = 0.0
        self.state_dict['f_s_water'] = 0.0
        self.state_dict['f_d_water'] = 0.0
        self.state_dict['fx_p'] = 0.0
        self.state_dict['fy_p'] = 0.0
        self.state_dict['m_air'] = 0.0
        self.state_dict['m_water'] = 0.0
        self.state_dict['mr'] = 0.0
        self.state_dict['my_p'] = 0.0



    def set_control(self, power, propeller_angle):

        # check for bounds
        if power > self.state_dict['power_max']:
            self.state_dict['power'] = self.state_dict['power_max']
        elif power < 0.0:
            self.state_dict['power'] = 0.0
        else:
            self.state_dict['power'] = power

        # check for bounds of the propeller angle
        if propeller_angle < self.state_dict['delta_max'][0]:
            self.state_dict['delta'] = self.state_dict['delta_max'][0]
        elif propeller_angle > self.state_dict['delta_max'][1]:
            self.state_dict['delta'] = self.state_dict['delta_max'][1]
        else:
            self.state_dict['delta'] = propeller_angle

    def get_aero_coeffs(self, x):
        """
        get the aerodynamic coefficients that act on the boat from the wind and the relative wind induced by motion

        :param x: relative flow angle [deg]
        :return:
            axial - coefficient of axial flow
            side - coefficient of lateratl flow
            moment - coefficent for induced moment
            normal side - coefficient for when flow is directly perpendicular
        """

        # drag coefficient
        cd = 0.195738 + 0.518615 * np.abs(x) - 0.496029 * x * x + 0.0941925 * np.abs(x) ** 3 + \
             1.86427 * np.sin(2.0 * np.pi * np.power(np.abs(x) / np.pi, 1.05)) * np.exp(
            -2.17281 * np.power(np.abs(x) - np.pi / 2.0, 2.0))

        # side force coefficient
        cs = np.sign(x) * (12.3722 - 15.453 * np.abs(x) + 6.0261 * np.abs(x * x) - 0.532325 * np.abs(x) ** 3) * \
             np.sin(np.abs(x)) * np.exp(-1.68668 * np.power(np.abs(x) - np.pi / 2.0, 2.0))  # - np.sign(x)*0.1

        # yaw coefficient
        cy = np.sign(x) * (0.710204 - 0.297196 * np.abs(x) + 0.0857296 * np.abs(x * x)) * np.sin(
            np.pi * 2.0 * np.power(np.abs(x) / np.pi, 1.05))

        # perpendicualr side force coefficinet
        cr = 0.904313

        return cd, cs, cy, cr

    def get_hydro_coeffs(self, x):

        # drag coefficient
        cd = 0.245219 - 0.93044 * np.abs(x) + 0.745752 * np.abs(x * x) - 0.15915 * np.power(np.abs(x), 3.0) + \
             2.79188 * np.sin(2.0 * np.abs(x)) * np.exp(-1.05667 * np.power(np.abs(x) - np.pi / 2.0, 2.0))

        # side force coefficient
        cs = np.sign(x) * (0.115554 + 3.09423 * np.abs(x) - 0.984923 * x * x) * np.sin(np.abs(x))

        # yaw coefficient
        cy = np.sign(x) * (0.322986 + 0.317964 * np.abs(x) - 0.1021844 * x * x) * np.sin(2.0 * np.abs(x))

        #
        cr = 2.545759

        return cd, cs, cy, cr

    def get_moment_hull(self, cr, vy):
        """

        :param cr: side force at phi_eff at 90[deg]. The flow velocity in the normal direction of the hull while spinning
        :param vy: effective transvers velocity
        :return:
        """

        l = self.state_dict['hull_length']
        omega = self.state_dict['psi_dot']
        # if np.abs(omega) < 1e-3:
        #    omega = 0.0
        alpha = cr * self.state_dict['area_water'] * self.state_dict['density_water'] / (l / 2.0)

        # forward porition
        mrf = l * l * alpha / 192.0 * (3 * l * l * omega * omega + 16.0 * l * omega * vy + 24.0 * vy * vy)

        if np.abs(vy) >= np.abs(omega * l / 2.0):
            mrb = -l * l * alpha / 192.0 * (3.0 * l * l * omega * omega - 16.0 * l * omega * vy + 24.0 * vy * vy)
        else:
            mrb = alpha / (192.0 * omega * omega) * (
                        np.power(l * omega - 2.0 * vy, 3.0) * (3.0 * l * omega + 2 * vy) - 16.0 * np.power(vy, 4.0))

        mr = mrf + mrb

        # adjust the direction of the moment based on the rate of rotation
        if self.state_dict['psi_dot'] < 0:
            mr = np.abs(mr)
        else:
            mr = -np.abs(mr)

        return mr

    def step_euler(self, time):

        # correct power if there is no fuel
        if self.state_dict['fuel'] <= 0.0:
            self.state_dict['power'] = 0.0
            self.state_dict['thrust'] = 0.0
        else:
            # get the thrust the propeller is currently outputing
            self.state_dict['thrust'] = self.calc_thrust([self.state_dict['v_xp'], self.state_dict['v_yp']],
                                                         self.state_dict['psi_dot'])

        # get the forces and moments of the boat. save them for telemetry later
        self.calc_forces_and_moments(self.state_dict['thrust'])

        fx_p = self.state_dict['f_d_air'] + self.state_dict['f_d_water'] + self.state_dict['fx_p']
        delta_xp = self.state_dict['v_xp'] * self.state_dict['delta_t'] + 0.5 * fx_p / self.state_dict['mass'] * \
                   self.state_dict['delta_t'] * self.state_dict['delta_t']

        fy_p = self.state_dict['f_s_air'] + self.state_dict['f_s_water'] + self.state_dict['fy_p']
        delta_yp = self.state_dict['v_yp'] * self.state_dict['delta_t'] + 0.5 * fy_p / self.state_dict['mass'] * \
                   self.state_dict['delta_t'] * self.state_dict['delta_t']

        mom = self.state_dict['m_air'] + self.state_dict['m_water'] + self.state_dict['my_p'] + self.state_dict['mr']
        delta_psi = self.state_dict['psi_dot'] * self.state_dict['delta_t'] + 0.5 * mom * (
                    self.state_dict['hull_length'] / 2.0) / self.state_dict['moi'] * self.state_dict['delta_t'] * \
                    self.state_dict['delta_t']

        self.state_dict['psi'] = self.state_dict['psi'] + delta_psi
        if self.state_dict['psi'] > 2.0 * np.pi:
            self.state_dict['psi'] -= 2.0 * np.pi
        elif self.state_dict['psi'] < 0.0:
            self.state_dict['psi'] += 2.0 * np.pi

        # convert change in position to global frame
        delta_x = delta_xp * np.cos(-self.state_dict['psi']) + delta_yp * np.sin(-self.state_dict['psi'])
        delta_y = -delta_xp * np.sin(-self.state_dict['psi']) + delta_yp * np.cos(-self.state_dict['psi'])

        self.state_dict['x_pos'] = self.state_dict['x_pos'] + delta_x
        self.state_dict['y_pos'] = self.state_dict['y_pos'] + delta_y

        self.state_dict['v_xp'] = self.state_dict['v_xp'] + fx_p / self.state_dict['mass'] * self.state_dict['delta_t']
        self.state_dict['v_yp'] = self.state_dict['v_yp'] + fy_p / self.state_dict['mass'] * self.state_dict['delta_t']
        self.state_dict['psi_dot'] = self.state_dict['psi_dot'] + mom / self.state_dict['moi'] * self.state_dict[
            'delta_t']

        self.state_dict['v_x'] = self.state_dict['v_xp'] * np.cos(-self.state_dict['psi']) + self.state_dict[
            'v_yp'] * np.sin(-self.state_dict['psi'])
        self.state_dict['v_y'] = -self.state_dict['v_xp'] * np.sin(-self.state_dict['psi']) + self.state_dict[
            'v_yp'] * np.cos(-self.state_dict['psi'])

        self.state_dict['acc_xp'] = fx_p / self.state_dict['mass']
        self.state_dict['acc_yp'] = fy_p / self.state_dict['mass']
        self.state_dict['psi_double_dot'] = mom / self.state_dict['moi']

        # convert acceleration to global reference plane
        self.state_dict['acc_x'] = self.state_dict['acc_xp'] * np.cos(-self.state_dict['psi']) + self.state_dict[
            'acc_yp'] * np.sin(-self.state_dict['psi'])
        self.state_dict['acc_y'] = -self.state_dict['acc_xp'] * np.sin(-self.state_dict['psi']) + self.state_dict[
            'acc_yp'] * np.cos(-self.state_dict['psi'])

        # calculate the fuel used in the simulation
        fuel_used = self.state_dict['power'] * self.state_dict['bsfc'] * self.state_dict['delta_t']  # [kg of fuel]
        self.state_dict['fuel'] -= fuel_used

        if self.state_dict['fuel'] < 0:
            self.state_dict['fuel'] = 0.0

        self.state_dict['time'] = time

        # log telemetry
        tmp_df = pd.DataFrame([self.state_dict])
        self.telemetry = pd.concat([self.telemetry, tmp_df], ignore_index=True)

    def calc_forces_and_moments(self, thrust):
        """
        given the state of the boat and the current thrust level, determine the forces abd moments from the air, water,
        and the propeller. axial, transverse, and moments are found for both air and water. An additional moment is
        found from the rotational component of the boat. Forces and moments induced on the boat from the propeller
        are also found.

        :param state: a vector of x position, local x velocity, y position, local y velocity, angle, angular velocity
        :param thrust: the amount of thrust [N]. This is also a function of state
        :return:
            f_d_air - axial force from the air [N]
            f_s_air - lateral force from the air [N]
            m_air - moment induced from the air [N-m]
            f_d_hydro - axial force from the water [N]
            f_s_hydro - lateral force from the water [N]
            m_hydro - moment induced from the water [N-m]
            fx_p - axial force from the propeller [N]
            fy_p - lateral force from the propeller [N]
            my_p - moment induced from the propeller [N-m]
            mr - moment induced by the boat rotating in the water [N-m]
        """

        rho_air = self.state_dict['density_air']
        rho_water = self.state_dict['density_water']

        # in global coordinates
        v_eff_air = self.state_dict['v_wind'] - [self.state_dict['v_x'], self.state_dict['v_y']]
        self.state_dict['v_x_eff_air'] = v_eff_air[0]
        self.state_dict['v_y_eff_air'] = v_eff_air[1]
        rot_angle = -self.state_dict['psi']

        rot_mat = [[np.cos(rot_angle), -np.sin(rot_angle)],
                   [np.sin(rot_angle), np.cos(rot_angle)]]
        rot_mat = np.reshape(rot_mat, (2, 2))
        v_eff_air_local = np.matmul(rot_mat, v_eff_air)

        # air forces and moments
        phi_eff_air_local = np.arctan2(v_eff_air_local[1], v_eff_air_local[0])
        self.state_dict['psi_eff_air'] = phi_eff_air_local
        v_eff_air_local_mag = np.linalg.norm(v_eff_air_local)

        cd_aero, cs_aero, cy_aero, cr_aero = self.get_aero_coeffs(phi_eff_air_local)

        # f_d_air = -np.sign(v_eff_air_local[0])* 0.5*rho_air*v_eff_air_local_mag *v_eff_air_local_mag*self.state_dict['area_air']*cd_aero
        f_d_air = 0.5 * rho_air * v_eff_air_local_mag * v_eff_air_local_mag * self.state_dict['area_air'] * cd_aero
        # f_s_air = -np.sign(v_eff_air_local[1])*0.5 * rho_air * v_eff_air_local_mag * v_eff_air_local_mag * self.state_dict['area_air']* cs_aero
        f_s_air = 0.5 * rho_air * v_eff_air_local_mag * v_eff_air_local_mag * self.state_dict['area_air'] * cs_aero

        # TODO check this
        m_air = -0.5 * cy_aero * self.state_dict['area_air'] * self.state_dict[
            'hull_length'] * rho_air * v_eff_air_local_mag * v_eff_air_local_mag

        # --------------------------------------------------------------------------------------------------------------
        # water forces
        v_eff_water = self.state_dict['v_current'] - [self.state_dict['v_x'], self.state_dict['v_y']]
        self.state_dict['v_x_eff_water'] = v_eff_water[0]
        self.state_dict['v_y_eff_water'] = v_eff_water[1]
        rot_angle = -self.state_dict['psi']
        rot_mat = [[np.cos(rot_angle), -np.sin(rot_angle)],
                   [np.sin(rot_angle), np.cos(rot_angle)]]
        rot_mat = np.reshape(rot_mat, (2, 2))
        v_eff_water_local = np.matmul(rot_mat, v_eff_water)
        v_eff_water_local_mag = np.linalg.norm(v_eff_water_local)

        phi_eff_water_local = np.arctan2(v_eff_water_local[1], v_eff_water_local[0])
        self.state_dict['psi_eff_water'] = phi_eff_water_local
        cd_hydro, cs_hydro, cy_hydro, cr_hydro = self.get_hydro_coeffs(phi_eff_water_local)

        # f_d_hydro = -np.sign(v_eff_water_local[0])*0.5*rho_water*v_eff_water_local_mag *v_eff_water_local_mag *self.state_dict['area_water']*cd_hydro
        f_d_hydro = 0.5 * rho_water * v_eff_water_local_mag * v_eff_water_local_mag * \
                    self.state_dict['area_water'] * cd_hydro
        # f_s_hydro = -np.sign(v_eff_water_local[1])*0.5 * rho_water * v_eff_water_local_mag  * v_eff_water_local_mag  * self.state_dict['area_water'] * cs_hydro
        f_s_hydro = 0.5 * rho_water * v_eff_water_local_mag * v_eff_water_local_mag * \
                    self.state_dict['area_water'] * cs_hydro

        m_hydro = -0.5 * cy_hydro * self.state_dict['area_water'] * self.state_dict[
            'hull_length'] * rho_air * v_eff_water_local_mag * v_eff_water_local_mag

        mr = self.get_moment_hull(cr_hydro, self.state_dict['v_yp'])

        # propulsion forces

        fx_p = thrust * np.cos(self.state_dict['delta'])
        fy_p = thrust * np.sin(self.state_dict['delta'])
        my_p = -fy_p * self.state_dict['hull_length'] / 2.0

        self.state_dict['f_d_air'] = f_d_air
        self.state_dict['f_s_air'] = f_s_air
        self.state_dict['m_air'] = m_air
        self.state_dict['f_d_water'] = f_d_hydro
        self.state_dict['f_s_water'] = f_s_hydro
        self.state_dict['m_water'] = m_hydro
        self.state_dict['fx_p'] = fx_p
        self.state_dict['fy_p'] = fy_p
        self.state_dict['my_p'] = my_p
        self.state_dict['mr'] = mr

    def get_alpha(self, v_local, psi_dot):
        """
        determines the angle of attack or incidence of the flow travelling across the propeller disk

        :param v_local: velocity of the boat in its local reference frame [m/s]
        :param psi_dot: the rotational velocity of the boat [rad/s]
        :return: the angle of attack of the flow to the propeller disk
        """

        # lateral velocity induced at the propeller from the boat yawing
        v_rot = self.state_dict['hull_length'] / 2.0 * psi_dot
        v_local = np.array(v_local) * -1.0

        propeller_axial = np.array(
            [np.cos(self.state_dict['delta'] + np.pi / 2.0), np.sin(self.state_dict['delta'] + np.pi / 2.0)])
        alpha = self.get_angle_between_vectors(propeller_axial, [v_local[0], v_rot + v_local[1]], True)

        self.state_dict['alpha'] = alpha

        return alpha

    def get_angle_between_vectors(self, v1, v2, keep_sign):

        if keep_sign:
            # the sign the angle matters
            angle = np.arctan2(v2[1] * v1[0] - v2[0] * v1[1], v1[0] * v2[0] + v1[1] * v2[1])
        else:
            # the sign should be ignored
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        return angle

    def calc_thrust(self, v_local, psi_dot):
        """
        calculates the thrust delivered to the propeller based on the controlled power

        :param v_local: the velocity of the boat in its local reference frame
        :param psi_dot: the angular velocity of boat
        :return: the current thrust output of the propeller given a fixed power level
        """

        v_mag = np.linalg.norm(v_local)

        alpha_d = self.get_alpha(v_local, psi_dot)

        # Line search for inflecion point
        eps = 1e-3

        step = 0.1
        v_guess = 0.0
        f_eval_old = self.thrust_helper(v_guess, v_mag, alpha_d, self.state_dict['power'],
                                        self.state_dict['density_water'], self.state_dict['prop_area'])
        early_break = False
        while np.abs(step) > eps:

            v_guess += step
            f_eval_new = self.thrust_helper(v_guess, v_mag, alpha_d, self.state_dict['power'],
                                            self.state_dict['density_water'], self.state_dict['prop_area'])

            if f_eval_new < f_eval_old:
                # inflection point has been passed
                v_guess -= 2.0 * step
                step /= 10.0

            if np.sign(f_eval_new) != np.sign(f_eval_old):
                # a root has been crossed, and the loop can prematureing be broken
                early_break = True
                break

            f_eval_old = f_eval_new

        if early_break:
            # use bisect to solve for root as it has cross the x-axis

            a = 0
            fa = self.thrust_helper(a, v_mag, alpha_d, self.state_dict['power'], self.state_dict['density_water'],
                                    self.state_dict['prop_area'])
            b = v_guess
            error = 1.0
            eps = 1e-6
            while error > eps:

                c = (a + b) / 2.0
                fc = self.thrust_helper(c, v_mag, alpha_d, self.state_dict['power'], self.state_dict['density_water'],
                                        self.state_dict['prop_area'])

                if np.sign(fa) == np.sign(fc):
                    a = c
                    fa = fc
                else:
                    b = c

                error = (b - a) / 2.0

            v_induced = c

        else:
            # inflection point has not reached x axis

            # golden section search for root bounds
            # line search for bounds
            v_tmp = 1e-6  # initial guess
            k = 1
            GOLDEN_RATIO = 1.61803
            guess = [v_tmp]
            error = self.thrust_helper(v_tmp, v_mag, alpha_d, self.state_dict['power'],
                                       self.state_dict['density_water'], self.state_dict['prop_area'])
            errors = [error]
            delta = 0.1
            while True:

                v_tmp = v_tmp + delta * np.power(GOLDEN_RATIO, k)
                guess.append(v_tmp)
                error = self.thrust_helper(v_tmp, v_mag, alpha_d, self.state_dict['power'],
                                           self.state_dict['density_water'], self.state_dict['prop_area'])
                errors.append(error)

                if np.sign(errors[k - 1]) != np.sign(errors[k]):
                    break

                if v_tmp > 100.0:
                    # the search has failed. Assume induced velocity is zero
                    break

                k += 1

            # bisect search for root
            error = 1.0
            eps = 1e-3
            a = guess[k - 1]
            b = guess[k]
            fa = self.thrust_helper(a, v_mag, alpha_d, self.state_dict['power'], self.state_dict['density_water'],
                                    self.state_dict['prop_area'])
            while error > eps:

                c = (a + b) / 2.0
                fc = self.thrust_helper(c, v_mag, alpha_d, self.state_dict['power'], self.state_dict['density_water'],
                                        self.state_dict['prop_area'])

                if np.sign(fc) == np.sign(fa):
                    a = c
                    fa = fc
                else:
                    b = c

                error = (b - a) / 2.0

            v_induced = c

        thrust = self.state_dict['power'] / (
                    v_mag + v_induced)  # 2.0*self.state_dict['density_water']*self.state_dict['prop_area']*v_induced*v_induced*np.sign(v_induced)

        return thrust

    def thrust_helper(self, v, v0, alpha_d, power, rho, area):

        vh = self.vh_calc(power, rho, area)

        p1 = np.power(v / vh, 4.0) + 2.0 * (v0 / vh) * np.power(v / vh, 3.0) * np.sin(alpha_d) + np.power(v0 / vh,
                                                                                                          2.0) * np.power(
            v / vh, 2.0)
        p2 = np.power((v0 * np.sin(alpha_d) + v) / vh, 2.0)

        return p1 * p2 - 1

    def vh_calc(self, power, rho, area):
        return np.power(power / (2.0 * rho * area), 1.0 / 3.0)

    def update_eq(self, v, v_mag, alpha_d):

        g1 = np.sqrt(v_mag * v_mag + 2.0 * v_mag * np.sin(alpha_d) * v + v * v)
        g2 = v_mag * np.sin(alpha_d) + v

        v_new = v - (v - self.state_dict['power'] * self.state_dict['fom'] / (
                    2.0 * self.state_dict['density_water'] * self.state_dict['prop_area']) / (g1 * g2)) / (
                            1.0 + self.state_dict['power'] * self.state_dict['fom'] / (
                                2.0 * self.state_dict['density_water'] * self.state_dict['prop_area']) * (
                                        (g2 * g2 + g1 * g1 / g1) / (g1 * g2) ** 2))

        return v_new

    @staticmethod
    def get_default(delta_t):
        """
        a basic boat for use in training. Based on the Grm model

        :param delta_t: time step the step function operates over
        :return: a built boat that has the properties already populated
        """
        name = 'river_boat'
        area_air = 15  # [m^2]
        area_water = 2.5  # [m^2]
        delta = 0
        delta_max = [-np.pi / 2.0, np.pi / 2.0]
        density_air = 1.225
        density_water = 998
        fom = 0.75  # figure of merit

        hull_len = 10.0
        hull_width = 2.5
        mass = 5000  # [kg]
        moi = 23000  # [kg m^2]
        psi = 0
        power = 9500  # [watt]
        power_max = 9500  # [watt]
        prop_diam = 0.25  # [m]
        fuel_capacity = 2.0  # [kg]
        fuel = 10.0  # [kg]
        # bsfc = 5.0e-8  # [kg/w-s] this is the realistic value
        bsfc = 5.0e-7  # [kg/w-s] this is the inefficient value for use
        # bsfc = 1.05e-5  # [kg/w-s] this is the inefficient value for use
        rb = RiverBoat(name, area_air, area_water, bsfc, delta, delta_max, delta_t, density_air, density_water, fom,
                       fuel, fuel_capacity
                       , hull_len, hull_width, mass, moi, power, power_max, psi, prop_diam)

        return rb