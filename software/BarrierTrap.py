import sys
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time

class BarrierTrap():
    '''
    Special double well that has a barrier between the two wells

    params [dict]: Parameters that define the potential
                   x_LW, y_LW = trap_center
                   x_RW, y_RW = target_center
                   k_wh = obstacle_height
                   r_a = channel_r_a
                   r_b = channel_r_b
                   k_ws = squeeze
                   1/k_wc = inv_obstacle_coverage

                   # The following entries are not relevant to the paper as the values we used for them simplify out of the function
                   trap_width = 1
                   trap_exp = 2
                   target_width = 1
                   target_exp = 2
                   
    '''
    def __init__(self, params=None):
        self.params = params
        self.params['inv_obstacle_coverage'] = 1/params['obstacle_coverage']
        self.params['channel_r_a'] = self.params['obstacle_dist'] - params['obstacle_width']
        self.params['channel_r_b'] = self.params['obstacle_dist'] + params['obstacle_width']

    def energy(self, X, input_mode='Traj'):
        '''
        Takes two coordinates (X=[x_1,x_2]) and returns the energy 
        '''
        if input_mode == 'Traj':
            x = X[:,0]
            y = X[:,1]
        if input_mode == 'Batch':
            x = X[:,:,0]
            y = X[:,:,1]
        
        # Calculate contribution of trap
        x_minus_trap_center = x - self.params['trap_center'][0]
        y_minus_trap_center = y - self.params['trap_center'][1]
        R = np.sqrt(x_minus_trap_center**2 + y_minus_trap_center**2)
        E_trap = ((R/self.params['trap_width'])**self.params['trap_exp']) 
        
        # Calculate contribution of target
        x_minus_target_center = x - self.params['target_center'][0]
        y_minus_target_center = y - self.params['target_center'][1]
        R = np.sqrt(x_minus_target_center**2 + y_minus_target_center**2)
        E_target = ((R/self.params['target_width'])**self.params['target_exp'])
        
        # Calculate contribution of obstactle
        ## Project to polar coordinates about arc center
        x_minus_arc_center = x - self.params['arc_center'][0]
        y_minus_arc_center = y - self.params['arc_center'][1]
        theta = np.arctan2(y_minus_arc_center, x_minus_arc_center)
        r = np.sqrt(x_minus_arc_center**2 + y_minus_arc_center**2)
        E_obstacle_theta = 1/(1+np.exp(-((theta-math.pi/self.params['inv_obstacle_coverage'])*self.params['squeeze']))) + 1/(1+np.exp(((theta+math.pi/self.params['inv_obstacle_coverage'])*self.params['squeeze']))) - 1
        E_obstacle_r = 1/(1+np.exp(-((r-self.params['channel_r_a'] )*self.params['squeeze'])))  + 1/(1+np.exp(((r-self.params['channel_r_b'] )*self.params['squeeze']))) - 1
        E_obstacle = -(E_obstacle_theta * E_obstacle_r) * self.params['obstacle_height']
        
        return E_obstacle + (E_trap * E_target)

    def batch_gradient(self, Trajs, mode, dx=0.001):
        '''
        Returns the gradient of the energy of the paths passed to it. 

        Trajs [numpy array]: Array containing the path(s)
        mode ['numpy' or 'TF']: Should this function use numpy or tensorflow operations. Use 'TF' when calling this as part of network training. 
        dX [float]: Delta used for numerical differentiation
        '''
        traj_shape = np.shape(Trajs)
        dX = np.tile([dx,0], (traj_shape[0], traj_shape[1], 1))
        dY = np.tile([0,dx], (traj_shape[0], traj_shape[1], 1))
        
        grad_Ux = (self.energy(Trajs+dX, input_mode='Batch') - self.energy(Trajs-dX, input_mode='Batch'))/(2*dx)
        grad_Uy = (self.energy(Trajs+dY, input_mode='Batch') - self.energy(Trajs-dY, input_mode='Batch'))/(2*dx)

        if mode == 'numpy':
            grad_U = np.stack((grad_Ux, grad_Uy), axis=2)
        elif mode == 'TF':
            grad_U = tf.stack((grad_Ux, grad_Uy), axis=2)
        
        return grad_U
   
    def plot_energy_surface(self, x_lim=2):
        '''
        Generates a contour plot showing the energy landscape

        x_lim [float]: Defines xmax, xmin, ymax, ymin of the plot. 
        '''
        plt.rcParams['figure.figsize'] = 12, 10
        cmax =55

        # Calculate Energies for Main Plot
        x_1s = np.linspace(-x_lim,x_lim, 100)
        X1grid, X2grid = np.meshgrid(x_1s, x_1s) # x_2s = x_1s
        Xs = np.vstack([X1grid.flatten(), X2grid.flatten()]).T
        energies = self.energy(Xs)
        energies = energies.reshape((100,100))
        energies = np.minimum(energies,cmax)          

        # Plot Main
        plt.contourf(X1grid, X2grid, energies, 100, cmap='jet', cmax=1)
        cbar = plt.colorbar()
        cbar.set_label('Energy / kT')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$', labelpad=-10)
        # Plot Basin
        plot_reactive_well(self, color='green', pe_lw=0.5, pe_alph=0.4, pe_col='green', lw=4)
        plot_reactive_well(self, color='white', mode='f')

        # Calculate energy at key locations
        # key_locs = np.array([[-1,0],[0,0],[1,0]])
        # key_energies = self.energy(key_locs)
        # adj = np.array([0.1, 0.05])
        # for key_loc, key_energy in zip(key_locs, key_energies):
        #     plt.text(*(key_loc-adj), str(key_energy), color='white', fontsize=30)
        plt.show()
