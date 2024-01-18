import sys
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time

class TwoChan_DoubleWell():
    '''
    Contains functions relevant to two channel double well model.

    params [dict]: Parameters that define potential. k_BH is 'TM' and the maximum energy of the target region is given by 'target'.  
    '''
    def __init__(self, params):
        self.params = params

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
        
        E = self.energy_xy(x,y)
        return E
    
    def energy_xy(self, x, y):
        '''
        Takes xy and returns energy
        '''
        E = 9*((x**4) - (2)*(y**2) + (y**4) + (78/37)*(x**2)*((y**2)-1) + (1/90)*y + 1.11097 + self.params['TM']*np.exp(-(x**2+y**2)/0.1))
        return E

    def batch_gradient(self, X, mode, dX=0.00001):
        '''
        Returns the gradient of the energy of the paths passed to it. 

        X [numpy array]: Array containing the path(s)
        mode ['numpy' or 'TF']: Should this function use numpy or tensorflow operations. Use 'TF' when calling this as part of network training. 
        dX [float]: Delta used for numerical differentiation
        '''
        x = X[:,:,0]
        y = X[:,:,1]
        
        x_pl_dX = x + dX
        x_mi_dX = x - dX
        x_pl_Energy = self.energy_xy(x_pl_dX, y)
        x_mi_Energy = self.energy_xy(x_mi_dX, y)
        grad_Ux = (x_pl_Energy - x_mi_Energy)/(2*dX)

        y_pl_dX = y + dX
        y_mi_dX = y - dX
        y_pl_Energy = self.energy_xy(x, y_pl_dX)
        y_mi_Energy = self.energy_xy(x, y_mi_dX)
        grad_Uy = (y_pl_Energy - y_mi_Energy)/(2*dX)

        if mode == 'numpy':
            grad_U = np.stack((grad_Ux, grad_Uy), axis=2)
        elif mode == 'TF':
            grad_U = tf.stack((grad_Ux, grad_Uy), axis=2)
        
        return grad_U

    def plot_free_energy(self):
        '''
        Plots the changes in free energy with x_1
        '''
        x_1s = np.linspace(-2,2, 200)[:,None]
        x_2s = np.zeros((x_1s.size,1))
        Xs = np.hstack((x_1s, x_2s))
        energies = self.energy(Xs)
        
        import matplotlib.pyplot as plt
        ax = plt.gca()
        ax.set_xlabel('$x_1$ / a.u.')
        ax.set_ylabel('Energy / kT')
        ax.set_ylim(energies.min()-2, energies[int(energies.size/2)]+2.5)
        
        return ax.plot(x_1s, energies, linewidth=3, color='black')      
   
    def plot_energy_surface(self, x_lim=2, cmax=15, sf=2):
        '''
        Generates a contour plot showing the energy landscape

        x_lim [float]: Defines xmax, xmin, ymax, ymin of the plot. 
        cmax [float]: Maximum energy value displayed on color plot.
        sf [int]: Number of sig figs used for energy values displayed in the plot.
        '''
        plt.rcParams['figure.figsize'] = 18, 12

        # Calculate Energies for Main Plot
        x_1s = np.linspace(-x_lim,x_lim, 100)
        X1grid, X2grid = np.meshgrid(x_1s, x_1s) # x_2s = x_1s
        Xs = np.vstack([X1grid.flatten(), X2grid.flatten()]).T
        energies = self.energy(Xs)
        energies = energies.reshape((100,100))
        energies = np.minimum(energies,cmax)          
        ##### Calculate Energies for Reactive Basin Outline #####
        alt_x_1s = np.linspace(0,x_lim, 100)
        x_2s = np.linspace(-x_lim,x_lim, 100)
        alt_X1grid, alt_X2grid = np.meshgrid(alt_x_1s, x_2s) 
        alt_Xs = np.vstack([alt_X1grid.flatten(), alt_X2grid.flatten()]).T
        alt_energies = self.energy(alt_Xs)
        alt_energies = alt_energies.reshape((100,100))

        # Plot Main
        plt.contourf(X1grid, X2grid, energies, 50, cmap='jet', vmax=cmax)
        cbar = plt.colorbar()
        cbar.set_label('Energy / kT')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$', labelpad=-10)
        # Plot Basin
        plt.contour(alt_X1grid, alt_X2grid, alt_energies, levels=[1*self.params['target']], colors='white', linewidths=5, linestyles='--')

        # Calculate energy at key locations
        #vert 
        num_locs = 100
        key_locs = np.linspace(-x_lim, 0, num_locs)
        key_locs_2 = np.linspace(0, x_lim, num_locs)
        zeros = np.zeros(num_locs)
        vert_locs = np.stack([zeros, key_locs]).T
        vert_2_locs = np.stack([zeros, key_locs_2]).T
        hori_locs = np.stack([key_locs, zeros]).T
        vert_energies = self.energy(vert_locs)
        hori_energies = self.energy(hori_locs)
        vert_2_energies = self.energy(vert_2_locs)
        
        min_vert_energy_index = np.argmin(vert_energies)
        min_hori_energy_index = np.argmin(hori_energies)
        min_vert_2_energy_index = np.argmin(vert_2_energies)
        min_vert_energy = vert_energies[min_vert_energy_index]
        min_hori_energy = hori_energies[min_hori_energy_index]
        min_vert_2_energy = vert_2_energies[min_vert_2_energy_index]
        min_vert_loc = vert_locs[min_vert_energy_index]
        min_hori_loc = hori_locs[min_hori_energy_index]      
        min_vert_2_loc = vert_2_locs[min_vert_2_energy_index]
        
        key_locs = [min_vert_loc, min_hori_loc, [0,0], min_vert_2_loc]
        key_energies = [min_vert_energy, min_hori_energy, self.energy_xy(0,0), min_vert_2_energy]
        
        adj = np.array([0.1, 0.05])
        for key_loc, key_energy in zip(key_locs, key_energies):
            plt.text(*(key_loc-adj), str(np.round(key_energy, sf)), color='white', fontsize=30)
        plt.show()
        
        return key_locs
    
