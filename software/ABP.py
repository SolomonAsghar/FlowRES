import numpy as np
from tqdm import tqdm

def SinCos(Phi):
    return np.concatenate((np.cos(Phi), np.sin(Phi)), axis=1)

def gradient(X, params, dX=0.001):
    x = X[:,0]
    y = X[:,1]
    
    x_pl_dX = x + dX
    x_mi_dX = x - dX
    x_pl_Energy = ((params['k_x']*((x_pl_dX**2) - (params['x_0']**2))**2) + ((params['k_y']/2)*(y**2))) * params['energy_scale']
    x_mi_Energy = ((params['k_x']*((x_mi_dX**2) - (params['x_0']**2))**2) + ((params['k_y']/2)*(y**2))) * params['energy_scale']
    grad_Ux = (x_pl_Energy - x_mi_Energy)/(2*dX)

    y_pl_dX = y + dX
    y_mi_dX = y - dX
    y_pl_Energy = ((params['k_x']*((x**2) - (params['x_0']**2))**2) + ((params['k_y']/2)*(y_pl_dX**2))) * params['energy_scale']
    y_mi_Energy = ((params['k_x']*((x**2) - (params['x_0']**2))**2) + ((params['k_y']/2)*(y_mi_dX**2))) * params['energy_scale']
    grad_Uy = (y_pl_Energy - y_mi_Energy)/(2*dX)

    grad_U = np.stack((grad_Ux, grad_Uy), axis=1)
    
    return grad_U

def ABP(num_traj, traj_len, start_point, tran_coeff, rot_coeff, act_coeff, pot_coeff, params):
    # Vector holding the configuration of each trajectory
    R = np.tile(start_point[:2], (num_traj, 1))
    PHI = np.tile(start_point[2:], (num_traj, 1))

    # Generate Noise in Advance 
    R_noise = np.random.normal(0, tran_coeff, size=(traj_len, num_traj,2))
    PHI_noise = np.random.vonmises(0, 1/(rot_coeff**2), size=(traj_len,num_traj,1))

    # Array to hold generated trajs
    Trajs = np.empty(shape=(num_traj, traj_len, 3)) 

    for i in tqdm(range(traj_len)):
        ####### Potential #######
        dU = gradient(R, params)
        R = R + act_coeff*SinCos(PHI) - pot_coeff*dU + R_noise[i]

        PHI = PHI + PHI_noise[i]
        
        ######### Save ##########
        Trajs[:,i,:2] = R
        Trajs[:,i,2:] = PHI
        #########################

    return Trajs