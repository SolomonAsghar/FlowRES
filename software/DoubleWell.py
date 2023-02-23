import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class DoubleWell():
    '''
    Provides energy function for a 2D double well system; used to calculate J_kl
    '''
    params_default = {'x_0' :  1.0,
                      'k_x' :  6.0,
                      'k_y' : 20.0}

    def __init__(self, params=None, energy_scale=1,
                 xmin=None, xmax=None, ymin=None, ymax=None):
        if params == None:
            params = self.__class__.params_default
        self.params = params
        
        self.energy_scale = energy_scale
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

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
        
        E = ((self.params['k_x']*((x**2) - (self.params['x_0']**2))**2) + ((self.params['k_y']/2)*(y**2))) * self.energy_scale
        
        return E

    def batch_gradient(self, X, mode, dX=0.001):
        x = X[:,:,0]
        y = X[:,:,1]
        
        x_pl_dX = x + dX
        x_mi_dX = x - dX
        x_pl_Energy = ((self.params['k_x']*((x_pl_dX**2) - (self.params['x_0']**2))**2) + ((self.params['k_y']/2)*(y**2))) * self.energy_scale
        x_mi_Energy = ((self.params['k_x']*((x_mi_dX**2) - (self.params['x_0']**2))**2) + ((self.params['k_y']/2)*(y**2))) * self.energy_scale
        grad_Ux = (x_pl_Energy - x_mi_Energy)/(2*dX)

        y_pl_dX = y + dX
        y_mi_dX = y - dX
        y_pl_Energy = ((self.params['k_x']*((x**2) - (self.params['x_0']**2))**2) + ((self.params['k_y']/2)*(y_pl_dX**2))) * self.energy_scale
        y_mi_Energy = ((self.params['k_x']*((x**2) - (self.params['x_0']**2))**2) + ((self.params['k_y']/2)*(y_mi_dX**2))) * self.energy_scale
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
   

    def plot_energy_surface(self, col_bar=False, x_lim=2):
        '''
        Generates a contour plot showing the energy landscape
        '''
        plt.rcParams['figure.figsize'] = 12, 8

        x_1s = np.linspace(-x_lim,x_lim, 100)
        X1grid, X2grid = np.meshgrid(x_1s, x_1s) # x_2s = x_1s
        Xs = np.vstack([X1grid.flatten(), X2grid.flatten()]).T
        energies = self.energy(Xs)
        energies = energies.reshape((100,100))
        energies = np.minimum(energies, 26.0)          #sets maximum possible energy to 10.0 (replaces any values >10 with 10)
        
        if col_bar==False:
            ax = plt.gca()
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$', labelpad=-10)

            return ax.contourf(X1grid, X2grid, energies, 50, cmap='jet', vmax=26)
        else:
            plt.contourf(X1grid, X2grid, energies, 50, cmap='jet', vmax=26)
            cbar = plt.colorbar()
            cbar.set_label('Energy / kT')
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$', labelpad=-10)