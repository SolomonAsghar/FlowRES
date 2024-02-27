import numpy as np
import scipy.spatial
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D

def JSD(Hist, val_Hist):
    Counts_x = np.sum(Hist, axis=0)
    Counts_y = np.sum(Hist, axis=1)
    val_Counts_x = np.sum(val_Hist, axis=0)
    val_Counts_y = np.sum(Hist, axis=1)
    JS_x = scipy.spatial.distance.jensenshannon(Counts_x, val_Counts_x, axis=(0))
    JS_y = scipy.spatial.distance.jensenshannon(Counts_y, val_Counts_y, axis=(0))
    return (JS_x + JS_y)/2


def compare_JSDS(FlowRES_JSD_vs_Proposals, Direct_Integration_JSD_vs_Proposals, plot_deviation_from_target=True):
    y_label = 'JSD with Validation Distribution'
    if plot_deviation_from_target:
        y_label = 'Devaiation from Target JSD'
        Target_JSD = Direct_Integration_JSD_vs_Proposals[1,-1]
        Direct_Integration_JSD_vs_Proposals[1] = Direct_Integration_JSD_vs_Proposals[1] - Target_JSD
        FlowRES_JSD_vs_Proposals[1] = FlowRES_JSD_vs_Proposals[1] - Target_JSD
    
    plt.rcParams['figure.figsize'] = 12,8
    plt.rcParams['font.size'] = 30

    red = '#3C91E6'
    blue = '#BF0603'

    lw_1 = 3
    lw_2 = 5
    plt.plot(*FlowRES_JSD_vs_Proposals, color=blue, linewidth=lw_1)
    plt.plot(*Direct_Integration_JSD_vs_Proposals, color=red, linewidth=lw_1)
    plt.xlabel('Num. Paths Proposed')
    plt.ylabel(y_label)
    plt.xscale('log')
    plt.xlim(Direct_Integration_JSD_vs_Proposals[0,0])
    plt.ylim(0)

    legend_elements = [Line2D([0], [0], color=blue, linewidth=4, label='FlowRES'),
                       Line2D([0], [0], color=red, linewidth=4, label='Langevin')]
    plt.legend(handles = legend_elements, ncol=1, fontsize=25, frameon=False)

    plt.show()
    
    
plasma = matplotlib.colormaps['plasma']
plasma_new_colors = plasma(np.linspace(0,1,256)**0.5)
plasma_2 = ListedColormap(plasma_new_colors)

height_ratios = [0.75,2]
width_ratios = [2,0.75]
figsize = 8*(2.75/2)
plt.rcParams['figure.figsize'] = figsize,figsize*1.033
plt.rcParams['font.size'] = 25
gs = gridspec.GridSpec(2, 2, height_ratios=height_ratios, width_ratios=width_ratios) 
gs.update(wspace=0.0, hspace=0.0)

def Get_Density(counts, bins):
    '''
    Convert a count to a density
    '''
    return counts / (sum(counts) * np.diff(bins))


def Hist2D(Trajs, bins):
    Trajs = np.concatenate(Trajs)
    Trajs_x = Trajs[:,0].flatten()
    Trajs_y = Trajs[:,1].flatten()
    return np.histogram2d(Trajs_x, Trajs_y, bins, density=True)[0].T


def clip_reac(Trajs, potential, include_last=True, return_lens=False):
    '''
    Return a mask where only valid trajs are True
    '''
    energies = potential.energy(Trajs, input_mode='Batch')
    energy_mask = energies < potential.params['target']
    right_mask = Trajs[:,:,0] > 0
    r_e_mask = np.logical_and(energy_mask,right_mask)
    not_r_e_mask = np.logical_not(r_e_mask)
    unreactive_mask = np.all(not_r_e_mask, axis=1)
    reactive_mask = np.logical_not(unreactive_mask)
    
    reac_Trajs = Trajs[reactive_mask]
    arg_reactive = np.argwhere(r_e_mask[reactive_mask])
    ind_of_first_reac = np.unique(arg_reactive[:,0], return_index=True)[1]
    timestep_of_first_reac = arg_reactive[ind_of_first_reac,1]

    clipped_Trajs = []
    if include_last is True:
        timestep_of_first_reac += 1
    for traj, first in zip(reac_Trajs,timestep_of_first_reac):
        clipped_Trajs.append(traj[:first])
    
    if return_lens is False:
        return clipped_Trajs
    else:
        return clipped_Trajs, timestep_of_first_reac+1

    
def Compare_Hists(NF_Hist_2D, Target_Hist_2D, bins, potential, text, Add_Scale_Bar=True, cmap=plasma_2, path=None):
    fsz = 33
    
    plt.rcParams['figure.figsize'] = 12, 12*1.03
    barrier_height = int(potential.params['k_BH'])
    main, ax1 = Plot_pc_Hist(NF_Hist_2D,  bins, c1='#3C91E6', lw=5, plot_target_well=True, potential=potential,
                 barrier_height=barrier_height, Add_Scale_Bar=Add_Scale_Bar, cmap=cmap, text=text)
    if path:
        ax1.plot(*path, color='white', linewidth=3)
    Plot_pc_Hist(Target_Hist_2D, bins, plot_xy=False, c1='#FDAB33', linestyle= (0, (10,5)), alph=0.0, lw=6, Add_Scale_Bar=False, zorder=1, cmap=cmap, text=text)
    cbaxes = inset_axes(ax1,
                        width="80%",  # width = 10% of parent_bbox width
                        height="3.0%",  # height : 50%
                        loc='lower center')
    cb = plt.colorbar(main, cax=cbaxes, ticklocation='top', 
                      label=r'$\hat{\rho}(x,y)$', orientation="horizontal")
    cbarticks = list(np.arange(*cbaxes.get_xlim(), 0.5))
    highest_p1 = np.floor((cbaxes.get_xlim()[1])*10)/10
    if not cbarticks[-1] >= highest_p1:
        cbarticks += [highest_p1]
    cbartick_labels = [str(cbarticks[0])] + ['' for i in range(len(cbarticks)-2)] + [str(cbarticks[-1])]
    cb.set_ticks(cbarticks, labels=cbartick_labels, fontsize=fsz)
    cbaxes.tick_params(axis='both', colors='white')
    cb.outline.set_edgecolor('white')
    cb.set_label(r'$\hat{\rho}(x,y)$', color='white', labelpad=-15, fontsize=fsz)
    plt.show()
    
    
def Plot_pc_Hist(Hist_2D, bins, gs=gs, figsize=15, c1='blue', alph = 0.21, plot_xy=True, linestyle='solid', lw=5.0,
                 plot_target_well=False, potential=None, barrier_height=None, Add_Scale_Bar=True, zorder=1, cmap=None, text=None):
    fsz = 33
    plt.rcParams['font.size'] = 35
    ypad = 20
    spine_color = '#0C0786'
    
    Counts_x = np.sum(Hist_2D, axis=0)
    Counts_y = np.sum(Hist_2D, axis=1)
    
    Counts_x = Get_Density(Counts_x, bins)
    Counts_y = Get_Density(Counts_y, bins)
    
    ax0 = plt.subplot(gs[0,0])
    ax1 = plt.subplot(gs[1,0])
    ax2 = plt.subplot(gs[1,1])

    # x 
    ax0.stairs(Counts_x, bins, color=c1, linewidth=lw, linestyle=linestyle, zorder=zorder)
    ax0.stairs(Counts_x, bins, color=c1, fill=True, alpha=alph, zorder=zorder)
    ax0.axes.get_xaxis().set_visible(False)
    ax0.set_xlim(bins[0],bins[-1])
    ax0.yaxis.label.set_color(spine_color)
    ax0.tick_params(axis='y', colors=spine_color)
    ax0.set_ylabel(r'$\rho(x)$', rotation=0, labelpad=-50, y=1.01)
    yticks = list(np.arange(*ax0.get_ylim(), 0.5)[1:])
    ytick_labels = ['' for i in range(len(yticks[:-1]))] + yticks[-1:]
    ax0.yaxis.set_ticks(yticks, labels=ytick_labels)
    
    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax0.spines['left'].set_position('center')
    # Eliminate upper and right axes
    ax0.spines['right'].set_color('none')
    ax0.spines['top'].set_color('none')
    ax0.spines['left'].set_linewidth(2.0)
    ax0.spines['left'].set_color(spine_color)
    ax0.yaxis.set_tick_params(width=2.0)
        
    # y 
    ax2.stairs(Counts_y, bins, color=c1, linewidth=lw, orientation='horizontal',  linestyle=linestyle, zorder=zorder)
    ax2.stairs(Counts_y, bins, color=c1, fill=True, alpha=alph, orientation='horizontal', zorder=zorder)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_ylim(bins[0],bins[-1])
    ax2.xaxis.label.set_color(spine_color)
    ax2.tick_params(axis='x', colors=spine_color)
    
    ax2.set_xlabel(r'$\rho(y)$', labelpad=-65, x=1.3)
    ax2.xaxis.set_ticks([1])
    
    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax2.spines['bottom'].set_position('center')
    # Eliminate upper and right axes
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.spines['bottom'].set_linewidth(2.0)
    ax2.spines['bottom'].set_color(spine_color)
    ax2.xaxis.set_tick_params(width=2.0)
    xticks = list(np.arange(*ax2.get_xlim(), 0.5)[1:])
    xtick_labels = ['' for i in range(len(xticks[:-1]))] + xticks[-1:]
    ax2.xaxis.set_ticks(xticks, labels=xtick_labels)
    
    if plot_xy is True:
        # x y
        main = ax1.imshow(Hist_2D, interpolation='nearest', origin='lower',
                   extent=[bins[0], bins[-1], bins[0], bins[-1]], cmap=cmap)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        if plot_target_well is True:
            x_lim=1.75
            alt_x_1s = np.linspace(0,x_lim, 100)
            x_2s = np.linspace(-x_lim,x_lim, 100)
            alt_X1grid, alt_X2grid = np.meshgrid(alt_x_1s, x_2s) 
            alt_Xs = np.vstack([alt_X1grid.flatten(), alt_X2grid.flatten()]).T
            alt_energies = potential.energy(alt_Xs)
            alt_energies = alt_energies.reshape((100,100))
            ax1.contour(alt_X1grid, alt_X2grid, alt_energies, levels=[potential.params['target']], colors='white', linewidths=5, linestyles='--')
            ax1.plot(-1,0, marker='X', markerfacecolor='white', markeredgecolor='black', linewidth=1, markersize=20, zorder=100)
                
        # Add scale bar #
        if Add_Scale_Bar is True:
            rect = patches.Rectangle((-1.85, 1.80), 1, 0.05, linewidth=1, edgecolor='None', facecolor='white')
            ax1.add_patch(rect)      
            ax1.text(-1.35, 1.725, "1", color='white', fontsize=fsz,
                     horizontalalignment='center', verticalalignment='top')
        
        ax1.text(1.85, 1.85, text, color='white',
                 horizontalalignment='right', verticalalignment='top')
        
        legend_elements = [Line2D([0], [0], color='#3C91E6', linewidth=6, label='FlowRES'),
                           Line2D([0], [0], color='#FDAB33', linewidth=6, linestyle='--', label='Validation')]
        plt.legend(loc=(0.04,1.05), handles = legend_elements, ncol=1, fontsize=27.5, frameon=False)
        
        return main, ax1