import tensorflow as tf
import math 

class Wave_unit():
    '''
    Based on wavenet.

    Same instead of causal padding
    
    output_size (int): Number of output neurons
    n_type (char): Network type, Scaling network (s) or a Translation network (t)
    '''
    def __init__(self, dim, WaveParams):
        self.dim = dim
        self.num_filters = WaveParams['num_filters']
        self.kernel_size = WaveParams['kernel_size']
        self.num_dilated_conv_layers = WaveParams['num_dilated_conv_layers']
        self.pre_build_layers()

    def pre_build_layers(self):
        '''
        We need to instantiate each trainable layer we will use before the call function, otherwise each call will generate a new (randomly initialized) Wave_unit and the network will not be invertible.
        '''
        def dilated_conv_layer(dilation):
            filter_conv = tf.keras.layers.Conv1D(self.num_filters, kernel_size=self.kernel_size, padding='same', dilation_rate=dilation, kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='tanh')
            gate_conv = tf.keras.layers.Conv1D(self.num_filters, kernel_size=self.kernel_size, padding='same', dilation_rate=dilation, kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='sigmoid')

            skip_out = tf.keras.layers.Conv1D(self.num_filters, kernel_size=1,  kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')
            layer_out = tf.keras.layers.Conv1D(self.num_filters, kernel_size=1,  kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')

            return [filter_conv, gate_conv, layer_out, skip_out]

        self.conv = tf.keras.layers.Conv1D(self.num_filters, kernel_size=self.kernel_size, padding='same',  kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')

        self.dilation_unit_layers = []
        for layer in range(self.num_dilated_conv_layers):
            self.dilation_unit_layers.append(dilated_conv_layer(2**(layer+1)))

        self.conv_1x1_a = tf.keras.layers.Conv1D(self.num_filters, kernel_size=1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')
        self.conv_1x1_b = tf.keras.layers.Conv1D(self.dim, kernel_size=1, padding='same', kernel_initializer=tf.keras.initializers.Zeros()) # by default, the bias initialises to zeros
    

    def __call__(self, input_layer):
        ''' 
        Now we connect the layers
        '''
        def dilated_conv_layer(dilation_input, filter_conv, gate_conv, layer_out, skip_out):
            filter_conv = filter_conv(dilation_input)
            gate_conv = gate_conv(dilation_input)

            gated_activation = tf.keras.layers.multiply([filter_conv, gate_conv])

            skip_out = skip_out(gated_activation)
            layer_out = layer_out(gated_activation)

            layer_out = tf.keras.layers.Add()([layer_out, dilation_input])      # residual connection  

            return layer_out, skip_out

        conv = self.conv(input_layer)

        dilation_input = conv
        skip_connections = []
        for layer in range(self.num_dilated_conv_layers):
            filter_conv, gate_conv, layer_out, skip_out = self.dilation_unit_layers[layer]

            dilation_input, skip_out = dilated_conv_layer(dilation_input, filter_conv, gate_conv, layer_out, skip_out)
            skip_connections.append(skip_out)
        
        sum_skip = tf.keras.layers.Add()(skip_connections)
        relu_sum_skip = tf.keras.layers.ReLU()(sum_skip)
        conv_1x1_a = self.conv_1x1_a(relu_sum_skip)
        final_layer = self.conv_1x1_b(conv_1x1_a)

        return final_layer
    
    
    
class SplineReady_WaveUnit():
    '''
    Based on wavenet but same instead of causal padding.
    Outputs three tensors that may parameterise a RationalQuadraticSpline bijector
    '''
    def __init__(self, dim, curr_len, WaveParams, SplineParams):
        self.dim = dim
        self.curr_len = curr_len
        self.num_filters = WaveParams['num_filters']
        self.kernel_size = WaveParams['kernel_size']
        self.num_dilated_conv_layers = WaveParams['num_dilated_conv_layers']
        self.n_bins = SplineParams['n_bins']
        self.spline_width = SplineParams['spline_width']
        self.min_bin_size = SplineParams['min_bin_size']
        self.min_slope = SplineParams['min_slope']
        self.pre_build_layers()

    def pre_build_layers(self):
        '''
        We need to instantiate each trainable layer we will use before the call function, otherwise each call will generate a new (randomly initialized) Wave_unit and the network will not be invertible.
        '''
        def dilated_conv_layer(dilation):
            filter_conv = tf.keras.layers.Conv1D(self.num_filters, kernel_size=self.kernel_size, padding='same', dilation_rate=dilation, kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='tanh')
            gate_conv = tf.keras.layers.Conv1D(self.num_filters, kernel_size=self.kernel_size, padding='same', dilation_rate=dilation, kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='sigmoid')

            skip_out = tf.keras.layers.Conv1D(self.num_filters, kernel_size=1,  kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')
            layer_out = tf.keras.layers.Conv1D(self.num_filters, kernel_size=1,  kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')

            return [filter_conv, gate_conv, layer_out, skip_out]

        self.conv = tf.keras.layers.Conv1D(self.num_filters, kernel_size=self.kernel_size, padding='same',  kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')

        self.dilation_unit_layers = []
        for layer in range(self.num_dilated_conv_layers):
            self.dilation_unit_layers.append(dilated_conv_layer(2**(layer+1)))

        self.conv_1x1_a = tf.keras.layers.Conv1D(self.num_filters, kernel_size=1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')
        self.conv_1x1_b = tf.keras.layers.Conv1D(self.dim*(3*self.n_bins), kernel_size=1, padding='same',
                                                 kernel_initializer=tf.keras.initializers.Zeros(), bias_initializer=tf.keras.initializers.Constant(value=math.log(math.exp(1.0-self.min_slope)-1.0)))

    def __call__(self, input_layer):
        ''' 
        Now we connect the layers
        '''
        def dilated_conv_layer(dilation_input, filter_conv, gate_conv, layer_out, skip_out):
            filter_conv = filter_conv(dilation_input)
            gate_conv = gate_conv(dilation_input)

            gated_activation = tf.keras.layers.multiply([filter_conv, gate_conv])

            skip_out = skip_out(gated_activation)
            layer_out = layer_out(gated_activation)

            layer_out = tf.keras.layers.Add()([layer_out, dilation_input])      # residual connection  

            return layer_out, skip_out

        conv = self.conv(input_layer)

        dilation_input = conv
        skip_connections = []
        for layer in range(self.num_dilated_conv_layers):
            filter_conv, gate_conv, layer_out, skip_out = self.dilation_unit_layers[layer]

            dilation_input, skip_out = dilated_conv_layer(dilation_input, filter_conv, gate_conv, layer_out, skip_out)
            skip_connections.append(skip_out)
        
        sum_skip = tf.keras.layers.Add()(skip_connections)
        relu_sum_skip = tf.keras.layers.ReLU()(sum_skip)
        conv_1x1_a = self.conv_1x1_a(relu_sum_skip)
        conv_1x1_b = self.conv_1x1_b(conv_1x1_a)
        
        ### Seperate and prepare for Spline ###
        raw_bin_widths, raw_bin_heights, raw_slopes, raw_pt = tf.split(conv_1x1_b, [self.dim*self.n_bins, self.dim*self.n_bins, self.dim*(self.n_bins-1), self.dim*1], axis=-1)
        rehsaped_bin_widths = tf.keras.layers.Reshape([self.curr_len, self.dim, self.n_bins])(raw_bin_widths)
        rehsaped_bin_heights = tf.keras.layers.Reshape([self.curr_len, self.dim, self.n_bins])(raw_bin_heights)
        rehsaped_slopes = tf.keras.layers.Reshape([self.curr_len, self.dim, self.n_bins-1])(raw_slopes)
        rehsaped_pt = tf.keras.layers.Reshape([self.curr_len, self.dim])(raw_pt)

        #########################################
        bin_widths = tf.math.softmax(rehsaped_bin_widths, axis=-1) * (self.spline_width - self.n_bins*self.min_bin_size) + self.min_bin_size
        bin_heights = tf.math.softmax(rehsaped_bin_heights, axis=-1) * (self.spline_width - self.n_bins*self.min_bin_size) + self.min_bin_size
        slopes = tf.math.softplus(rehsaped_slopes) + self.min_slope
        pt = rehsaped_pt - math.log(math.exp(1.0-self.min_slope)-1.0)
        #########################################

        return bin_widths, bin_heights, slopes, pt