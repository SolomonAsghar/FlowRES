import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button


def suggest_maximum_weight(weights, n_bins=100, minimum_count=1, mode='last over'):
    '''
    Plot a histogram of weights with n_bins bins, and find the highest weight bin with at least minimum_count frequency 
    '''
    weights_histogram, weights_bins = np.histogram(weights, n_bins)
    weights_histogram_less_min = np.where(weights_histogram > 0, weights_histogram - 1, 0)
       
    if mode == 'last over':
        arg_over_min = np.nonzero(weights_histogram_less_min)[0]
        last_over_min = np.max(arg_over_min) + 1
        upper_edge = weights_bins[last_over_min]    
    if mode == 'first under':
        arg_under_min = np.where(weights_histogram_less_min == 0)
        first_under_min = np.min(arg_under_min) + 1
        upper_edge = weights_bins[first_under_min]    
        
    return upper_edge

def get_reac_mask(Positions, potential):
    '''
    Return a mask where only valid trajs are True
    '''
    energies = potential.energy(Positions, input_mode='Batch')
    energy_mask = energies < 2
    right_mask = Positions[:,:,0] > 0
    r_e_mask = np.logical_and(energy_mask,right_mask)
    not_r_e_mask = np.logical_not(r_e_mask)
    unreactive_mask = np.all(not_r_e_mask, axis=1)
    reactive_mask = np.logical_not(unreactive_mask)
    return reactive_mask

def Zero_Times(Times, dp=2):
    return ((Times - Times[0]) * 10**dp).astype('int') / 10**dp

def Get_Density(counts, bins):
    '''
    Convert a count to a density
    '''
    return counts / (sum(counts) * np.diff(bins))

def Calculate_Histograms(Trajs, weights=None, bins=None, only_reactive=False, potential=None, suggest_max_weight = False):
    '''
    Convert data into histogram in XY, X, and Y
    '''
    if only_reactive is True:
        reactive_mask = get_reac_mask(Trajs, potential)
        Trajs = Trajs[reactive_mask]
        if weights is not None:
            weights = weights[reactive_mask]
    
    if suggest_max_weight is True:
        lower, upper = 0, suggest_maximum_weight(weights)
        realistic_traj_indicies = np.where(np.logical_and(weights>lower, weights<upper))[0]
        weights = weights[realistic_traj_indicies]
        Trajs = Trajs[realistic_traj_indicies]
    
    if weights is not None:
        weights = np.tile(weights, reps=(np.shape(Trajs)[1],1)).flatten('F')
    
    Trajs_x = Trajs[:,:,0].flatten()
    Trajs_y = Trajs[:,:,1].flatten()
    Counts_2D = np.histogram2d(Trajs_x, Trajs_y, bins, weights=weights, density=True)[0].T
    
    Counts_x = np.sum(Counts_2D, axis=0)
    Counts_y = np.sum(Counts_2D, axis=1)

    D_Counts_x = Get_Density(Counts_x, bins)
    D_Counts_y = Get_Density(Counts_y, bins)
    
    return Counts_2D, D_Counts_x, D_Counts_y

def Plot_Histogram(gs, Counts_2D, D_Counts_x, D_Counts_y, bins,
                   D_Comp_x=None, D_Comp_y=None,
                   D_Comps_x_raw=None, D_Comps_y_raw=None,
                   Time=None, JSD=None, PA=None,
                   c1='blue', c2 = 'red', c3 = 'teal', alph = 0.21):
    '''
    Plot a histogram of the path density 
    '''
    
    # Setup 
    ax0 = plt.subplot(gs[0,0])
    ax0.cla()
    ax1 = plt.subplot(gs[1,0])
    ax1.cla()
    ax2 = plt.subplot(gs[1,1])
    ax2.cla()
    
    # Conditional plots
    if D_Comp_x is not None:
        ax0.stairs(D_Comp_x, bins, color=c2, linewidth=3)
        ax0.stairs(D_Comp_x, bins, color=c2, fill=True, alpha=alph)
        ax2.stairs(D_Comp_y, bins, color=c2, linewidth=3, orientation='horizontal')
        ax2.stairs(D_Comp_y, bins, color=c2, fill=True, alpha=alph, orientation='horizontal')
    if D_Comps_x_raw is not None:
        ax0.stairs(D_Comps_x_raw, bins, color=c3, linewidth=3)
        ax0.stairs(D_Comps_x_raw, bins, color=c3, fill=True, alpha=alph)
        ax2.stairs(D_Comps_y_raw, bins, color=c3, linewidth=3, orientation='horizontal')
        ax2.stairs(D_Comps_y_raw, bins, color=c3, fill=True, alpha=alph, orientation='horizontal')
    if Time is not None:
        time_info_text = 't = ' + str(Time) + 's'
        ax1.text(-1.90,-1.90, time_info_text, color='white', fontsize='x-large', fontweight='bold')
    if JSD is not None:
        time_info_text = 'JSD = ' + str(JSD) 
        ax1.text(0.9,-1.90, time_info_text, color='white', fontsize='x-large', fontweight='bold')
    if PA is not None:
        time_info_text = 'R. % = ' + str(PA) 
        ax1.text(-1.90, 1.80, time_info_text, color='white', fontsize='x-large', fontweight='bold')
    
    # x 
    ax0.stairs(D_Counts_x, bins, color=c1, linewidth=3)
    ax0.stairs(D_Counts_x, bins, color=c1, fill=True, alpha=alph)
    # ax0.axes.get_xaxis().set_visible(False)
    ax0.set_ylabel('$P$')
    ax0.set_ylim(0,1)
    ax0.set_xlim(bins[0],bins[-1])
    ax0.yaxis.label.set_color(c1)
    ax0.tick_params(axis='y', colors=c1)

    # x y
    ax1.imshow(Counts_2D, interpolation='nearest', origin='lower',
               extent=[bins[0], bins[-1], bins[0], bins[-1]], cmap='jet')
    ax1.set_xlabel('$x$',fontsize=20)
    ax1.set_ylabel('$y$',fontsize=20)

    # y the
    ax2.stairs(D_Counts_y, bins, color=c1, linewidth=3, orientation='horizontal')
    ax2.stairs(D_Counts_y, bins, color=c1, fill=True, alpha=alph, orientation='horizontal')
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_xlabel('$P$')
    ax2.set_xlim(0,1)
    ax2.set_ylim(bins[0],bins[-1])
    ax2.xaxis.label.set_color(c1)
    ax2.tick_params(axis='x', colors=c1)
    
def Plot_Relative_Entropy(gs, times, divergence, raw_divergence=None,
                          c1='blue', c3 = 'teal',
                          iteration=-1):
    '''
    Plot relative entropies vs time
    '''
    ax = plt.subplot(gs[5,:])
    ax.cla()
    if raw_divergence is not None:
        ax.plot(times, raw_divergence, color=c3)
    ax.plot(times, divergence, color=c1)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('J.S. Div.')
    ax.axvline(times[iteration], linestyle='--', color='red')
    
def Plot_Percentage_Reactive(gs, times, percentage_reactive,
                             c1='orange',
                             iteration=-1):
    '''
    Plot relative entropies vs time
    '''
    ax = plt.subplot(gs[7,:])
    ax.cla()
    ax.plot(times, percentage_reactive, color=c1)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('percentage reactive (%).')
    ax.axvline(times[iteration], linestyle='--', color='red')
    
def Compare_Iterations(Chains, Times=None, Comp_Trajs=None, Weights_Chain=None, Evaluate_Weights=False, num_bins=100, figsize=15,
                       only_reactive=False, potential=None, suggest_max_weight=False):
    '''
    An interactive plots for seeing the evolution of MCMC across iterations
    '''
    num_iters = np.shape(Chains)[1]
    Hists_2D = np.empty([num_iters, num_bins, num_bins])
    Hists_x = np.empty([num_iters, num_bins])
    Hists_y = np.empty([num_iters, num_bins])
    if (Weights_Chain is not None) and (Evaluate_Weights is True):
        Raw_Hists_2D = np.empty([num_iters, num_bins, num_bins])
        Raw_Hists_x = np.empty([num_iters, num_bins])
        Raw_Hists_y = np.empty([num_iters, num_bins])
        raw_JS_divergence = np.zeros(num_iters)
    else:
        Raw_Hists_x, Raw_Hists_y, raw_relative_entropies = [[None]*num_iters]*3
    Times = Zero_Times(Times)
    
    # Calculate Histograms and relative entropies
    bins = np.linspace(-2,2, num_bins+1)   
    JS_divergence = np.zeros(num_iters)
    num_chains = np.shape(Chains)[0]
    percentage_reactive = np.zeros(num_iters)
    
    if Comp_Trajs is not None:
        Comps_2D, Comp_xs, Comp_ys = Calculate_Histograms(Comp_Trajs, bins=bins)
    else:
        Comp_xs, Comp_ys = None, None
    
    for i in range(num_iters):
        if Weights_Chain is not None: # Consider the weights if we have to
            Hists_2D[i], Hists_x[i], Hists_y[i] = Calculate_Histograms(Chains[:,i], Weights_Chain[:,i], bins, only_reactive=only_reactive, potential=potential, suggest_max_weight=suggest_max_weight)
            if Evaluate_Weights is not None: # Store unweighted if we have to
                Raw_Hists_2D[i], Raw_Hists_x[i], Raw_Hists_y[i] = Calculate_Histograms(Chains[:,i], bins=bins, only_reactive=only_reactive, potential=potential)
                raw_JS_divergence[i] = np.sum(scipy.spatial.distance.jensenshannon(Raw_Hists_2D[i].flatten(), Comps_2D.flatten()))
        else:
            Hists_2D[i], Hists_x[i], Hists_y[i] = Calculate_Histograms(Chains[:,i], bins=bins, only_reactive=only_reactive, potential=potential)
        # Calculate relative entropy of this 
        JS_divergence[i] = np.sum(scipy.spatial.distance.jensenshannon(Hists_2D[i].flatten(), Comps_2D.flatten()))
        # Calculate what percentage of paths are reactive
        num_reactive = np.count_nonzero(get_reac_mask(Chains[:,i], potential))
        percentage_reactive[i] = 100 * (num_reactive/num_chains)
    # Plot Histograms    
    height_ratios = [1, 2, 0.25, 0.15, 0.1, 0.5, 0.1, 0.5]
    width_ratios = [2,1]
    plt.rcParams['figure.figsize'] = figsize, sum(height_ratios)*(figsize/sum(width_ratios))
    
    fig = plt.figure()
    # set height ratios for subplots
    gs = gridspec.GridSpec(8, 2, height_ratios=height_ratios, width_ratios=width_ratios) 
    gs.update(wspace=0.0, hspace=0.0)

    if Evaluate_Weights is False:
        Plot_Histogram(gs, Hists_2D[-1], Hists_x[-1], Hists_y[-1], bins, Comp_xs, Comp_ys, Time=Times[-1], JSD=round(JS_divergence[-1], 3), PA=round(percentage_reactive[-1],2))
    else:
        Plot_Histogram(gs, Hists_2D[-1], Hists_x[-1], Hists_y[-1], bins, Comp_xs, Comp_ys, Raw_Hists_x[-1], Raw_Hists_y[-1], Time=Times[-1], JSD=round(JS_divergence[-1], 3), PA=round(percentage_reactive[-1],2))
        
    # Plot Relative Entropy
    Plot_Relative_Entropy(gs, Times, JS_divergence, raw_JS_divergence)
    Plot_Percentage_Reactive(gs, Times, percentage_reactive)

    # Make a horizontal slider to control the iteration shown.
    axfreq = fig.add_subplot(gs[3,:])
    iter_slider = Slider(ax=axfreq,
                         label='Iteration',
                         valmin=0,
                         valmax=num_iters-1,
                         valstep=1,
                         valinit=num_iters-1)
    
    def update(val):
        if Evaluate_Weights is False:
            Plot_Histogram(gs, Hists_2D[iter_slider.val], Hists_x[iter_slider.val], Hists_y[iter_slider.val], bins, Comp_xs, Comp_ys, Time=Times[iter_slider.val], JSD=round(JS_divergence[iter_slider.val], 3), PA=round(percentage_reactive[iter_slider.val],2))
        else:
            Plot_Histogram(gs, Hists_2D[iter_slider.val], Hists_x[iter_slider.val], Hists_y[iter_slider.val], bins, Comp_xs, Comp_ys, Raw_Hists_x[iter_slider.val], Raw_Hists_y[iter_slider.val], Time=Times[iter_slider.val], JSD=round(JS_divergence[iter_slider.val], 3), PA=round(percentage_reactive[iter_slider.val],2))
        Plot_Relative_Entropy(gs, Times, JS_divergence, raw_JS_divergence, iteration=iter_slider.val)
        Plot_Percentage_Reactive(gs, Times, percentage_reactive, iteration=iter_slider.val)
            
    iter_slider.on_changed(update)

    plt.show()