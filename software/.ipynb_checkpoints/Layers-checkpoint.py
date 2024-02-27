'''
Different invertible layers used by FlowRES
'''
import math
import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp
from util import connect, AngleToPeriodic

class AffineCoupling():
    '''
    A single affine coupling layer.

    transforms [network,network]: A list containing a network for S transformation and T transformation
    operand [string]: May be 'ang' (for angles) or 'pos' (for positions). Describes what this transformation operates upon.
    conditioner [string]: May be 'ang' (for angles) or 'pos' (for positions). Describes what this transformation is conditioned by.
    '''
    def __init__(self, transforms, operand=None, conditioner=None):
        self.operand = operand
        self.conditioner = conditioner
        if operand == conditioner:
            self.cross_couple = False
        else:
            self.cross_couple = True
        self.S = transforms[0]
        self.T = transforms[1]

    def XtoZ(self, X, X_op=None):
        '''
        Transforms X -> Z
        '''
        def l_exp(x):
            return tf.keras.backend.exp(x)
        def l_sum(x):
            # Don't sum over the samples axis
            return tf.keras.backend.sum(x, axis=(1,2))

        if self.cross_couple == False:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X
            x2 = X_op

        # Affine transform
        z1 = x1
        S_layer = connect(AngleToPeriodic(x1) if self.conditioner=='ang' else x1, self.S)
        T_layer = connect(AngleToPeriodic(x1) if self.conditioner=='ang' else x1, self.T)
        prodx = tf.keras.layers.Multiply()([x2, tf.keras.layers.Lambda(l_exp)(S_layer)])
        z2 = tf.keras.layers.Add()([prodx, T_layer])
         
        # Calculates log_Rxz = log|det(Jxz)|
        self.log_Rxz = tf.keras.layers.Lambda(l_sum)(S_layer)

        return [z1, z2]
    
    def ZtoX(self, Z, Z_op=None):
        '''
        Transforms Z -> X
        '''
        def l_neg_exp(z):
            return tf.keras.backend.exp(-z)
        def l_neg_sum(z):
            return tf.keras.backend.sum(-z, axis=(1,2))
        
        if self.cross_couple == False:
            z1 = Z[0]
            z2 = Z[1]
        else:
            z1 = Z
            z2 = Z_op
        
        # Inverse affine transform
        x1 = z1
        S_layer = connect(AngleToPeriodic(z1) if self.conditioner=='ang' else z1, self.S)
        T_layer = connect(AngleToPeriodic(z1) if self.conditioner=='ang' else z1, self.T)
        z2_less_Tz1 = tf.keras.layers.Subtract()([z2, T_layer])
        x2 = tf.keras.layers.Multiply()([z2_less_Tz1, tf.keras.layers.Lambda(l_neg_exp)(S_layer)])
        
        # Calculates log_Rzx = log|det(Jzx)|
        self.log_Rzx = tf.keras.layers.Lambda(l_neg_sum)(S_layer)
        
        return [x1, x2]
    

# class Spline():
#     '''
#     A rational quadratic spline transform layer, as in arXiv:1906.04032

#     ParameterNetwork (network): A network that outputs the parameters used by the Spline, namely bin_widths, bin_heights, knot_slopes and phase_translation. 
#     range_min (float): The smallest value the spline covers, below this the function is linear.
#     operand [string]: May be 'ang' (for angles) or 'pos' (for positions). Describes what this transformation operates upon.
#     conditioner [string]: May be 'ang' (for angles) or 'pos' (for positions). Describes what this transformation is conditioned by.
#     '''
#     def __init__(self, ParameterNetwork, range_min, operand=None, conditioner=None):
#         self.operand = operand
#         self.conditioner = conditioner
#         if operand == conditioner:
#             self.cross_couple = False
#         else:
#             self.cross_couple = True    
#         self.ParameterNetwork = ParameterNetwork
#         self.range_min = range_min

#     def XtoZ(self, X, X_op=None):
#         '''
#         Transforms X -> Z
#         '''
#         if self.cross_couple == False:
#             x1 = X[0]
#             x2 = X[1]
#         else:
#             x1 = X
#             x2 = X_op

#         # Transform
#         z1 = x1
#         bin_widths, bin_heights, knot_slopes, phase_translation = connect(AngleToPeriodic(x1) if self.conditioner=='ang' else x1,
#                                                                           [self.ParameterNetwork])
#         RQS = tfp.bijectors.RationalQuadraticSpline(bin_widths, bin_heights, knot_slopes, self.range_min)
#         z2_raw = RQS.forward(x2)
#         z2 = (z2_raw+phase_translation)
#         z2 = (z2 +math.pi)%(2*math.pi) -math.pi

#         # Calculates log_Rxz = log|det(Jxz)|
#         self.log_Rxz = tf.math.reduce_sum(RQS.forward_log_det_jacobian(x2), axis=(1,2))

#         return [z1, z2]
    
#     def ZtoX(self, Z, Z_op=None):
#         '''
#         Transforms Z -> X
#         '''        
#         if self.cross_couple == False:
#             z1 = Z[0]
#             z2 = Z[1]
#         else:
#             z1 = Z
#             z2 = Z_op
        
#         # Undo Transform
#         x1 = z1
#         bin_widths, bin_heights, knot_slopes, phase_translation = connect(AngleToPeriodic(z1) if self.conditioner=='ang' else z1,
#                                                                           [self.ParameterNetwork])
#         RQS = tfp.bijectors.RationalQuadraticSpline(bin_widths, bin_heights, knot_slopes, self.range_min)
#         z2_raw = (z2-phase_translation)
#         z2_raw = (z2_raw +math.pi)%(2*math.pi) -math.pi
#         x2 = RQS.inverse(z2_raw)

#         # Calculates log_Rzx = log|det(Jzx)|
#         self.log_Rzx = tf.math.reduce_sum(RQS.inverse_log_det_jacobian(z2_raw), axis=(1,2))
        
#         return [x1, x2]


class Inv_1x1(tf.keras.layers.Layer):
    '''
    A layer for invertible 1x1 convolutions
    Takes two inputs and stacks them, computes 1x1 convolution, unstacks.

    curr_len (int): the current lenght of each sample matrix
    curr_dim (int): the current width of each sample matrix
    operand [string]: May be 'ang' (for angles) or 'pos' (for positions). Describes what this transformation operates upon.
    '''
    def __init__(self, curr_len, curr_dim, operand=None):
        self.operand = operand
        self.conditioner = operand
        super(Inv_1x1, self).__init__()

        self.curr_len = curr_len
        self.curr_dim = curr_dim*2
        # Sample a random orthogonal matrix to initialise weights
        w_init = np.linalg.qr(np.random.randn(self.curr_dim, self.curr_dim))[0]
        self.W = tf.Variable(initial_value=w_init, dtype='float32', trainable=True)

    def call(self, input, direction):
        if direction == 'XtoZ':
            X = input
            W = tf.reshape(self.W, [1, self.curr_dim, self.curr_dim])
            Z = tf.nn.conv1d(X, filters=W, stride=1, padding='SAME')
            # self.log_Rxz_const = self.curr_len * tf.math.log(tf.math.abs(tf.linalg.det(self.W)))
            return Z
        if direction == 'ZtoX':
            Z = input
            value_inv_W = tf.linalg.inv(self.W)
            inv_W = tf.reshape(value_inv_W, [1, self.curr_dim, self.curr_dim])
            X = tf.nn.conv1d(Z, filters=inv_W, stride=1, padding='SAME')
            # self.log_Rzx_const = self.curr_len * tf.math.log(tf.math.abs(tf.linalg.det(value_inv_W)))
            return X
    
    def XtoZ(self, X):
        X = tf.concat([X[0], X[1]], axis=2)
        Z = self(X, direction=('XtoZ'))
        Z = tf.split(Z, num_or_size_splits=2, axis=2)
        return Z

    def ZtoX(self, Z):
        Z = tf.concat([Z[0], Z[1]], axis=2)
        X = self.call(Z, direction=('ZtoX'))
        X = tf.split(X, num_or_size_splits=2, axis=2)
        return X
    
class Inv_1x1_restore(Inv_1x1):
    '''
    A layer for reversing an invertible 1x1 convolution.
    Created for simplicity, but it's essentially the above with XtoZ and ZtoX flipped

    Inv_1x1_layer (Inv_1x1): The Inv_1x1 you would like to invert.
    '''
    def __init__(self, Inv_1x1_layer):
        super(Inv_1x1_restore, self).__init__(Inv_1x1_layer.curr_len, int(Inv_1x1_layer.curr_dim/2),
                                              Inv_1x1_layer.operand)
        self.W = Inv_1x1_layer.W
    
    def XtoZ(self, X):
        X = tf.concat([X[0], X[1]], axis=2)
        Z = self(X, direction=('ZtoX'))
        Z = tf.split(Z, num_or_size_splits=2, axis=2)
        return Z

    def ZtoX(self, Z):
        Z = tf.concat([Z[0], Z[1]], axis=2)
        X = self.call(Z, direction=('XtoZ'))
        X = tf.split(X, num_or_size_splits=2, axis=2)
        return X


class Alt_Channels():
    '''
    Splits data into seperate channels for X->Z, merges channels for Z->X
    A one dimensional version of checkerboard split used in RealNVP i.e. alternating pattern
    n.b.: named start as it is the first layer when going X->Z, however it will be the last when going Z->X.
    
    curr_len (int): Contains the lenght of the vector this will accept, used to create a key for merging and splitting 
    operand [string]: May be 'ang' or 'pos'. Describes what this transformation operates upon.
    conditioner [string]: May be 'ang' (for angles) or 'pos' (for positions). Describes what this transformation is conditioned by.
    ''' 
    def __init__(self, curr_len, operand=None):
        self.operand = operand
        self.conditioner = operand
        c_one = []
        c_two = []
        for config in range(curr_len):
            if config % 2 == 0:
                c_one.append(config)
            else:
                c_two.append(config)
        self.split_key = [c_one, c_two]
        self.merge_key = np.argsort([item for sublist in self.split_key for item in sublist])
    
    def XtoZ(self, X):
        return [tf.gather(X, i, axis=1) for i in self.split_key]
    
    def ZtoX(self, Z):
        unsorted = tf.keras.layers.concatenate(Z, axis=1)
        return tf.gather(unsorted, self.merge_key, axis=1)

    
class Squeeze():
    '''
    Creates a pair of equally shaped tensors using Alt_Channels and 'stacks' them.
    
    curr_len (int): Contains the lenght of the vector this will accept, used to create a key for merging and splitting 
    operand [string]: May be 'ang' or 'pos'. Describes what this transformation operates upon.
    '''     
    def __init__(self, curr_len, operand=None):
        self.operand = operand
        self.conditioner = operand
        self.Alt_Channels = Alt_Channels(curr_len)

    def XtoZ(self, X):
        x1_a, x1_b = self.Alt_Channels.XtoZ(X[0]) # 1,3,5,7.. -> [1,5,9..],[3,7,11...]
        x1 = tf.concat([x1_a, x1_b], axis=2) # stack along channel axis

        x2_a, x2_b = self.Alt_Channels.XtoZ(X[1])
        x2 = tf.concat([x2_a, x2_b], axis=2) # stack along channel axis

        return [x1,x2]

    def ZtoX(self, Z): # just the reverse of the above
        z2_a, z2_b = tf.split(Z[1], num_or_size_splits=2, axis=2)
        z2 = self.Alt_Channels.ZtoX([z2_a, z2_b])

        z1_a, z1_b = tf.split(Z[0], num_or_size_splits=2, axis=2)
        z1 = self.Alt_Channels.ZtoX([z1_a, z1_b])

        return [z1, z2]

class Concat():
    '''
    Concatenate a pair of tensors 
    operand [string]: May be 'ang' or 'pos'. Describes what this transformation operates upon.
    '''     
    def __init__(self, operand=None):
        self.operand = operand
        self.conditioner = operand
    def XtoZ(self, X):
        return tf.concat(X, axis=1) # stack along time axis
    def ZtoX(self, Z): # just the reverse of the above
        return tf.split(Z, num_or_size_splits=2, axis=1)

class Split():
    '''
    Concatenate a pair of tensors 
    operand [string]: May be 'ang' or 'pos'. Describes what this transformation operates upon.
    '''     
    def __init__(self, operand=None):
        self.operand = operand
        self.conditioner = operand
    def XtoZ(self, X):
        return tf.split(X, num_or_size_splits=2, axis=1)
    def ZtoX(self, Z): # just the reverse of the above
        return tf.concat(Z, axis=1) # stack along time axis

class Reshape():
    '''
    Reshapes the squeezed tensor back to its original shape, and ensures generated latent tensors get squeezed. 
    Needed as it makes concant of Exits easier; alternative is having a ZtoX network with floor((CBs-1)/(CBS_per_Exit))+1 inputs.

    curr_len (int): the current lenght of each sample matrix
    dim (int): the current width of each sample matrix
    final (bool): Whether or not this is the final Reshape in the flow
    operand [string]: May be 'ang' (for angles) or 'pos' (for positions). Describes what this transformation operates upon.
    ''' 
    def __init__(self, curr_len, dim, final=False, operand=None):
        self.operand = operand
        self.conditioner = operand
        self.curr_len = curr_len
        self.dim = dim
        self.final = final
    
    def XtoZ(self, X):
        if self.final == True:
            X = tf.concat(X, axis=1)
        return tf.keras.layers.Reshape((-1, self.dim))(X)
    
    def ZtoX(self, Z):
        Z = tf.keras.layers.Reshape((self.curr_len, -1))(Z)
        if self.final == True:
            Z = tf.split(Z, num_or_size_splits=2, axis=1) 
        return Z

class Exit():
    '''
    Used for early exist, when we need to factor out half the channels.

    curr_len (int): the current lenght of each sample matrix
    dim (int): the current width of each sample matrix
    operand [string]: May be 'ang' (for angles) or 'pos' (for positions). Describes what this transformation operates upon.
    '''
    def __init__(self, curr_len, dim, operand=None):
        self.operand = operand
        self.conditioner = operand
        self.Alt_Channels = Alt_Channels(curr_len)
        self.Reshape = Reshape(curr_len, dim)
    
    def XtoZ(self, X):
        z1 = self.Reshape.XtoZ(X[0]) # this part exits; reshape it so it matches the final input = initial input
        z2 = self.Alt_Channels.XtoZ(X[1]) # This continues on so we need to split it into two channels

        return [z1, z2] # note that this is really [z1, [z2_a, z2_b]]
    
    def ZtoX(self, Z): # Reverse of the above
        x2 = self.Alt_Channels.ZtoX(Z[1])
        x1 = self.Reshape.ZtoX(Z[0])

        return [x1, x2]
    
class Swap_Channels():
    '''
    Swaps two channels

    operand [string]: May be 'ang' (for angles) or 'pos' (for positions). Describes what this transformation operates upon.
    conditioner [string]: May be 'ang' (for angles) or 'pos' (for positions). Describes what this transformation is conditioned by.
    '''
    def __init__(self, operand=None):
        self.operand = operand
        self.conditioner = operand

    def XtoZ(self, X):
        return [X[1], X[0]]
    
    def ZtoX(self, Z):
        return [Z[1], Z[0]]

class Weave():
    '''
    The opposite of an Alternate Split, weaves two channels together in a way that inverts alt split.

    curr_len (int): Contains the lenght of the vector this will accept, used to create a key for merging and splitting 
    operand [string]: May be 'ang' or 'pos'. Describes what this transformation operates upon.
    conditioner [string]: May be 'ang' (for angles) or 'pos' (for positions). Describes what this transformation is conditioned by.
    '''
    def __init__(self, curr_len, mode, operand=None):
        self.operand = operand
        self.conditioner = operand
        self.mode = mode
        self.Alt_Channels = Alt_Channels(curr_len)

    def XtoZ(self, X):
        if self.mode == 'start':
            return self.Alt_Channels.ZtoX(X)
        if self.mode == 'end':
            return self.Alt_Channels.XtoZ(X)    

    def ZtoX(self, Z):
        if self.mode == 'start':
            return self.Alt_Channels.XtoZ(Z)
        if self.mode == 'end':
            return self.Alt_Channels.ZtoX(Z)
        
def Generate_restore_key(Shuffling_Layers, max_len, dims):
    ''' 
    Ensures that temporal ordering is consistent across the whole flow by calculating the changes made by all layers and inverting them.

    Shuffling_Layers (list of layers): A list of all the flows layers that effect temporal order.
    max_len (int): The desired max path lenght
    dims (int): The desired max path dimensions (excluding orientations)
    '''
    X = tf.range(0, max_len)
    X = [X for dim in range(dims)]
    X = tf.stack(X, axis=1)
    X = tf.expand_dims(X, axis=(0)) # dimension for samples

    output = []
    Z = None
    for layer in Shuffling_Layers:
        Z = layer.XtoZ(X)
        if not isinstance(layer, Exit): # for most layers do this
            X = Z
        else: # but if we're about to change scale, do this
            output.append(Z[0])
            X = Z[1]

    output.append(Z) # the last one will never need to be an Exit
    Z = tf.concat(output, axis=1) # The time axis

    ZtoX_key = Z[0,:,0]
    XtoZ_key = tf.argsort(ZtoX_key)

    return XtoZ_key, ZtoX_key
    
class Restore():
    '''
    Counter the shuffling that occours over the architecture, using the restore key.
    ''' 
    def __init__(self, XtoZ_key, ZtoX_key, operand=None):
        self.operand = operand
        self.conditioner = operand
        self.XtoZ_key = XtoZ_key
        self.ZtoX_key = ZtoX_key
    
    def XtoZ(self, X):
        return tf.gather(X, self.XtoZ_key, axis=1) 
    
    def ZtoX(self, Z):
        return tf.gather(Z, self.ZtoX_key, axis=1)
