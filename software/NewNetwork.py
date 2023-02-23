'''
WaveGlowNet for TPS of ABP. Function at end of file builds the network. 
'''
import numpy as np
import time 
import math
import sys
sys.path.append(r'/home/solomon.asghar/NF_TPS/software/')
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.utils import Progbar
tf.compat.v1.disable_eager_execution()

from util import connect, regularise, AngleToPeriodic
from Layers import *

########################################################## 
# The Network (including build, train etc.)
########################################################## 
class network():
    '''
    Assembles the layers(/models) passed to it by the function CreateBG into an invertible coupling flow 
    
    layers (list): List of invertible layers/models
    num_outputs (int): How many different sections the output is split up due to the MSA 
    max_len (int): The lenght of the trajectories handled
    dim (int): The dimensionality of the data. Defaults to 2 to ensure older notebooks work.
    start ([float, float]): The coordinates of the first point in all trajs
    '''
    def __init__(self, layers, num_outputs, max_len, dim, start, tran_coeff, rot_coeff, act_coeff, pot_coeff, potential_grad, network_mode):
        self.layers = layers[:-1]
        self.restore = layers[-1]     # handle this order restoring layer differently
        self.num_outputs = num_outputs
        self.max_len = max_len
        self.dim = dim
        self.start = start
        self.tran_coeff = tran_coeff
        self.rot_coeff = rot_coeff
        self.act_coeff = act_coeff
        self.pot_coeff = pot_coeff
        self.potential_grad = potential_grad
        self.network_mode = network_mode
        self.build_net()           
        
    def connect_XtoZ(self):
        '''
        Connects a sequence of RealNVP layers, from X to Z
        '''
        def ExitCheck(X, Z, output, layer):
            if not isinstance(layer, Exit): # for most layers do this
                X = Z
            else: # but if we're about to change scale, do this
                output.append(Z[0]) # send the first half of vector to the output
                X = Z[1] # keep operating on the other half
            return X, output

        # conditioner decides which input stream to use
        X_p = self.input_x_positions
        X_a = self.input_x_angles

        # operand decides which output list to append to
        output_p = [] 
        output_a = []
        Z_p = X_p # if no operation happens, just carry it forward
        Z_a = X_a

        for layer in self.layers:
            # Handle cases where we only operate on POS...
            if layer.conditioner == 'pos' and layer.operand == 'pos':
                Z_p = layer.XtoZ(X_p)
                X_p, output_p = ExitCheck(X_p, Z_p, output_p, layer)                
            # ... or ANG
            if layer.conditioner == 'ang' and layer.operand == 'ang':
                Z_a = layer.XtoZ(X_a)
                X_a, output_a = ExitCheck(X_a, Z_a, output_a, layer)
            # Handle cross couplings
            if layer.conditioner == 'ang' and layer.operand == 'pos':
                Z_a, Z_p = layer.XtoZ(X_a, X_p)
                X_a = Z_a
                X_p = Z_p # No need for exit check and only affine layers may cross couple
            if layer.conditioner == 'pos' and layer.operand == 'ang':
                Z_p, Z_a = layer.XtoZ(X_p, X_a)
                X_p = Z_p
                X_a = Z_a # No need for exit check and only affine layers may cross couple

        output_p.append(Z_p) # the last one will never need to be an Exit
        Z_p = tf.concat(output_p, axis=1) # The time axis
        output_a.append(Z_a) 
        Z_a = tf.concat(output_a, axis=1) 
        Z_a = self.restore.XtoZ(Z_a)

        return Z_p, Z_a

    def connect_ZtoX(self):
        '''
        Connects a sequence of RealNVP layers, from Z to X
        '''
        # conditioner decides which input stream to use
        Z_p = self.input_z_positions
        Z_a = self.input_z_angles
        Z_a = self.restore.ZtoX(Z_a)

        # Prepare the invert the MSA by calculating the size of each split up section
        if self.num_outputs > 1:
            size_splits = []
            for output in range(1, self.num_outputs):
                size_splits.append(int(self.max_len/(2**(output))))
            size_splits = size_splits + size_splits[-1:]
        else:
            size_splits = [int(self.max_len)]

        input_p = tf.split(Z_p, size_splits, axis=1)
        input_a = tf.split(Z_a, size_splits, axis=1)
        Z_p = input_p.pop(-1)
        Z_a = input_a.pop(-1)

        # Invert the MSA
        for layer in list(reversed(self.layers)):
            # Handle cases where we only operate on POS...
            if layer.conditioner == 'pos' and layer.operand == 'pos':
                if isinstance(layer, Exit):
                    Z_p = [input_p.pop(-1), Z_p]
                X_p = layer.ZtoX(Z_p)
                Z_p = X_p
            # ... or ANG
            if layer.conditioner == 'ang' and layer.operand == 'ang':
                if isinstance(layer, Exit):
                    Z_a = [input_a.pop(-1), Z_a]
                X_a = layer.ZtoX(Z_a)
                Z_a = X_a
            # Handle cross coupling, no exit check as only affine may cross couple
            if layer.conditioner == 'ang' and layer.operand == 'pos':
                X_a, X_p = layer.ZtoX(X_a, X_p)
                Z_a = X_a
                Z_p = X_p 
            if layer.conditioner == 'pos' and layer.operand == 'ang':
                X_p, X_a = layer.ZtoX(X_p, X_a)
                Zpa = X_p
                Z_a = X_a 

        return X_p, X_a
    
    # @tf.function
    def build_net(self): 
        # X -> Z
        self.input_x_positions = tf.keras.layers.Input(shape=(self.max_len,self.dim-1))#, name='True Positions')
        self.input_x_angles = tf.keras.layers.Input(shape=(self.max_len,1))#, name='True Angles')
        self.output_z_positions, self.output_z_angles = self.connect_XtoZ()
        
        # Z -> X
        self.input_z_positions = tf.keras.layers.Input(shape=(self.max_len,self.dim-1))#, name='Basis Positions')
        self.input_z_angles = tf.keras.layers.Input(shape=(self.max_len,1))#, name='Basis Angles')
        self.output_x_positions, self.output_x_angles = self.connect_ZtoX()
               
        self.net_XtoZ = tf.keras.Model([self.input_x_positions, self.input_x_angles], [self.output_z_positions, self.output_z_angles])
        self.net_ZtoX = tf.keras.Model([self.input_z_positions, self.input_z_angles], [self.output_x_positions, self.output_x_angles])


##########################################################
#################### DATA GEN ############################
    def Gaussian_Generator(self, size):
        '''
        Generates a Gaussian walk to be used as a prior. 
        '''
        while True: # or we only generate once
            Gaussian = np.random.normal(size=size)
            yield Gaussian
    
    def Angle_Walk_Generator(self, starts, size):
        '''
        Generates a Gaussian angle walk. 
        '''
        while True: # or we only generate once
            # Create rotational walk
            Ang_Increments = np.random.normal(size=(size[0], size[1], 1), scale=self.rot_coeff)
            Ang_Increments = np.concatenate((starts[:,:,2:], Ang_Increments), axis=1)
            Ang_Walk = np.cumsum(Ang_Increments, axis=1)
            Ang_Walk = Ang_Walk%(2*math.pi)
            yield Ang_Walk[:,1:]
##########################################################
#################### LOSSES ##############################
    @property
    # @tf.function
    def log_Rxz(self):
        '''
        Calculates log_Rxz = log|det(Jxz)| for the network, for the current batch
        '''
        var_log_Rxz_s = []
        const_log_Rxz_s = []
        for layer in self.layers:
            if hasattr(layer, 'log_Rxz'):          # required as some don't
                var_log_Rxz_s.append(layer.log_Rxz)
            if hasattr(layer, 'log_Rxz_const'):          # required as some operations have constant log_Rxz, i.e. there is only one value vs one per input
                const_log_Rxz_s.append(layer.log_Rxz_const)

        Sum_var_log_Rxz_s = tf.math.reduce_sum(var_log_Rxz_s, axis=0)   # avoid summing over the samples axis, so we can use this property to view log Rxz vales per traj
        return Sum_var_log_Rxz_s + tf.math.reduce_sum(const_log_Rxz_s)

    @property  
    # @tf.function
    def log_Rzx(self):
        '''
        Calculates log_Rzx = log|det(Jzx)| for the network, for the current batch
        '''
        var_log_Rzx_s = []
        const_log_Rzx_s = []
        for layer in self.layers:
            if hasattr(layer, 'log_Rzx'):          # required as some don't
                var_log_Rzx_s.append(layer.log_Rzx)
            if hasattr(layer, 'log_Rzx_const'):
                const_log_Rzx_s.append(layer.log_Rzx_const)
        
        Sum_var_log_Rzx_s = tf.math.reduce_sum(var_log_Rzx_s, axis=0)
        return Sum_var_log_Rzx_s + tf.math.reduce_sum(const_log_Rzx_s)

##########################################################
#################### DATA G ############################## 
    def transform_XtoZ(self, x_pos, x_ang):
        '''
        Apply the network to transform X onto Z
        '''
        # Snip off first steps
        x_pos = x_pos
        x_ang = x_ang
        z_pos, z_ang = self.net_XtoZ.predict([x_pos, x_ang])
        return z_pos, z_ang

    def transform_ZtoX(self, z_pos, z_ang):
        '''
        Apply the network to transform Z onto X
        '''
        starts = np.tile([-1,0,0], (len(z_pos),1,1)).astype('float32')
        x_pos, x_ang = self.net_ZtoX.predict([z_pos, z_ang])
        return np.concatenate((starts[:,:,:2], x_pos), axis=1), np.concatenate((starts[:,:,2:], x_ang), axis=1)

    def transform_ZtoX_weights(self, z_pos, z_ang):

        inputs_z = [self.input_z_positions, self.input_z_angles]
        outputs_x = [self.output_x_positions, self.output_x_angles, self.log_Rzx]
        net_ZtoX_log_Rzx = tf.keras.Model(inputs_z, outputs_x)
        
        x_pos, x_ang, log_Rzx = net_ZtoX_log_Rzx.predict([z_pos, z_ang])
        starts = np.tile([self.start], (len(z_pos),1,1)).astype('float32')

        # Prepare trajectories       
        Positions = np.concatenate((starts[:,:,:2], x_pos), axis=1)
        Angles = np.concatenate((starts[:,:,2:], x_ang), axis=1)
        #
        Latent_Positions = z_pos
        Latent_Angles = np.concatenate((starts[:,:,2:], z_ang), axis=1)

        # Calculate probs of latent vectors in generated latent distribution
        ## Prob of generated samples in target dist
        ### Calculate log prob of the positions
        Pos_Increments = Positions[:,1:] - Positions[:,:-1]
        Sin_Angles = np.sin(Angles)
        Cos_Angles = np.cos(Angles)
        grad_U = self.potential_grad(Positions[:,:-1], mode='numpy')
        Activity_Vector = np.concatenate([Cos_Angles[:,:-1], Sin_Angles[:,:-1]], axis=2)
        Translational_Noise = Pos_Increments + self.pot_coeff*grad_U - self.act_coeff*Activity_Vector
        Translational_Noise = Translational_Noise/self.tran_coeff
        pos_log_prob = - 0.5 * np.sum(Translational_Noise**2, axis=(1,2))
        ### Calculate log prob of the angles
        Ang_Increments = Angles[:,1:] - Angles[:,:-1]
        Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
        ang_log_prob = 1/(self.basis_rot_coeff**2) * np.sum(np.cos(Ang_Increments), axis=(1,2))
        ###
        log_prob_gens_in_target = pos_log_prob + ang_log_prob
        ### Add Rzx to get log probs of latent in generated latent
        log_prob_latent_in_gen_latent = log_prob_gens_in_target + log_Rzx

        # Calculate probs of latent vectors in true latent distribution
        ## log prob of the latent positions
        pos_latent_log_prob = - 0.5 * np.sum(Latent_Positions**2, axis=(1,2))
        ## log prob of the latent angles
        Ang_Increments = Angles[:,1:] - Angles[:,:-1]
        Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
        ang_latent_log_prob = 1/(self.basis_rot_coeff**2) * np.sum(np.cos(Ang_Increments), axis=(1,2))
        ###
        log_probs_latent_in_tru_latent = pos_latent_log_prob + ang_latent_log_prob
        ###

        # Calculate weights
        W_x = np.exp(log_prob_latent_in_gen_latent - log_probs_latent_in_tru_latent)

        return Positions, Angles, W_x
        
                 
########################################################## 
# Create, Save and Load Network
##########################################################
def CreateBG(Num_Layers, FLOWS_pre_LAYER,
             Pos_flows_per_Flow, Ang_flows_per_Flow, 
             CCs, Pos_CC_per_Flow, Ang_CC_per_Flow,
             Affine_WaveNet, Affine_WaveParams,
             Spline_WaveNet, Spline_WaveParams, SplineParams, Spline_range_min,
             start, tran_coeff, rot_coeff, act_coeff, pot_coeff,
             potential_grad,
             network_mode,
             max_len=400, dim=2):
    '''
    ### UPDATE THIS TEXT
    '''  
    # Work out the number of outputs 
    num_outputs = Num_Layers

    layers = []
    shuffling_layers = []

    curr_len = max_len
    curr_dim_pos = dim - 1
    curr_dim_ang = 1

    # divide the input into two channels
    layers.append(Alt_Channels(curr_len, operand='pos'))
    layers.append(Alt_Channels(curr_len, operand='ang'))
    shuffling_layers.append(Alt_Channels(curr_len))
    
    for L in range(Num_Layers): # for each layer
        
        curr_dim_pos = int(curr_dim_pos*2)
        curr_dim_ang = int(curr_dim_ang*2)
        curr_len = int(curr_len/2)
        layers.append(Squeeze(curr_len, operand='pos'))
        layers.append(Squeeze(curr_len, operand='ang'))
        shuffling_layers.append(Squeeze(curr_len))
        
        for FLOW in range (FLOWS_pre_LAYER):
            for Pos_flow in range(Pos_flows_per_Flow):    
                ## the two pos channels act on eachother
                Invertible_1x1 = Inv_1x1(curr_len, curr_dim_pos, operand='pos')
                layers.append(Invertible_1x1)
                S = Affine_WaveNet(curr_dim_pos, Affine_WaveParams); T = Affine_WaveNet(curr_dim_pos, Affine_WaveParams)
                layers.append(AffineCoupling([[S],[T]], operand='pos', conditioner='pos'))
                layers.append(Inv_1x1_restore(Invertible_1x1))
                if (Pos_flows_per_Flow%2==0) or Pos_flow<(Pos_flows_per_Flow-1):
                    layers.append(Swap_Channels(operand='pos'))

            for Ang_flow in range(Ang_flows_per_Flow):
                ## the two ang channels act on eachother
                Invertible_1x1 = Inv_1x1(curr_len, curr_dim_ang, operand='ang')
                layers.append(Invertible_1x1)
                ParameterNetwork = Spline_WaveNet(curr_dim_ang, int(curr_len/2), Spline_WaveParams, SplineParams)
                layers.append(Spline(ParameterNetwork, Spline_range_min, operand='ang', conditioner='ang'))
                layers.append(Inv_1x1_restore(Invertible_1x1))
                if (Ang_flows_per_Flow%2==0) or Ang_flow<(Ang_flows_per_Flow-1):
                    layers.append(Swap_Channels(operand='ang'))
                    shuffling_layers.append(Swap_Channels())
            
            ## cross couplings between angle and position
            layers.append(Weave(curr_len, 'start', 'pos'))
            layers.append(Weave(curr_len, 'start', 'ang'))
            for CC in range(CCs):
                for Pos_CC in range(Pos_CC_per_Flow):
                    # ang affects pos
                    S = Affine_WaveNet(curr_dim_pos, Affine_WaveParams); T = Affine_WaveNet(curr_dim_pos, Affine_WaveParams)
                    layers.append(AffineCoupling([[S],[T]], operand='pos', conditioner='ang'))
                for Ang_CC in range(Ang_CC_per_Flow):
                    # pos affects ang
                    ParameterNetwork = Spline_WaveNet(curr_dim_ang, curr_len, Spline_WaveParams, SplineParams)
                    layers.append(Spline(ParameterNetwork, Spline_range_min, operand='ang', conditioner='pos'))
            layers.append(Weave(curr_len, 'end', 'pos'))
            layers.append(Weave(curr_len, 'end', 'ang'))

        if not L == Num_Layers-1: # for all last layer, split
            curr_len = int(curr_len/2)
            layers.append(Exit(curr_len, dim-1, operand='pos'))
            layers.append(Exit(curr_len, 1, operand='ang'))
            shuffling_layers.append(Exit(curr_len, 1))

    layers.append(Reshape(curr_len, dim-1, final=True, operand='pos'))
    layers.append(Reshape(curr_len, 1, final=True, operand='ang'))
    shuffling_layers.append(Reshape(curr_len, 1, final=True))

    ### Restore the order of the Angles ###
    XtoZ_key, ZtoX_key = Generate_restore_key(shuffling_layers, max_len, 1)
    layers.append(Restore(XtoZ_key, ZtoX_key))
    #######################################

    print('... creating network')
    BG = network(layers, num_outputs, max_len, dim, start, tran_coeff, rot_coeff, act_coeff, pot_coeff, potential_grad, network_mode)
    print('Network created.')
    
    return BG 



def BG_build():
    print('NewNetwork 2.3')
##########################################################
##########################################################