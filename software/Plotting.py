import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

def PM_pi(Traj):
    return (Traj +math.pi)%(2*math.pi) -math.pi

def get_reac(Trajs, potential, weights=None):
    '''
    Return an array containing only valid trajectories
    '''
    Reac_Trajs = []
    if weights is not None:
        Reac_weights = []

    for i, traj in enumerate(Trajs):
        energies = potential.energy(traj)
        energy_mask = energies < 2

        right_mask = traj[:,0] > 0
        r_e_mask = np.logical_and(energy_mask,right_mask)

        if np.any(r_e_mask):    # if we have values within the target basin
            Reac_Trajs.append(traj)
            if weights is not None:
                Reac_weights.append(weights[i])

    if weights is None:
        return np.array(Reac_Trajs)
    else:
        return np.array(Reac_Trajs), np.array(Reac_weights)

def Plot_Hist(Trajs, Comp_Trajs=None, space='Config', n_bins=500, reactive=False, weights=None,
              evaluate_weights=False, weight_threshold_floor=0, weight_threshold_ceiling=30, potential=None):
    '''
    Plot a histogram in x,y
    '''
    # Params
    plt.rcParams['figure.figsize'] = 10, 13.3
    c1 = 'blue'
    c2 = 'red'
    c3 = 'teal'
    alph = 0.21

    if evaluate_weights is True:
        Trajs_Baseline = Trajs
        Trajs_Baseline = np.concatenate(Trajs_Baseline) # or we get peak at start

    if reactive:
        if weights is None:
            Trajs = get_reac(Trajs, potential)
        else:
            Trajs, weights = get_reac(Trajs, potential=potential, weights=weights)
            # remove unrealistic trajs
            realistic_traj_indicies = np.where(np.logical_and(weights>weight_threshold_floor, weights<weight_threshold_ceiling))[0]
            weights = weights[realistic_traj_indicies]
            Trajs = Trajs[realistic_traj_indicies]

    if weights is not None:
        # remove unrealistic trajs
        realistic_traj_indicies = np.where(np.logical_and(weights>weight_threshold_floor, weights<weight_threshold_ceiling))[0]
        weights = weights[realistic_traj_indicies]
        Trajs = Trajs[realistic_traj_indicies]
        weights = np.tile(weights, reps=(np.shape(Trajs)[1],1)).T
        weights = np.concatenate(weights)        
    Trajs = np.concatenate(Trajs) # or we get peak at start

    if Comp_Trajs is not None:
        if reactive:
            Comp_Trajs = get_reac(Comp_Trajs, potential)
        Comp_Trajs = np.concatenate(Comp_Trajs[:,1:]) # or we get peak at start
    
    if space == 'Config':
        bins = np.linspace(-2,2, n_bins)
    elif space == 'Latent':
        bins = np.linspace(-4,4, n_bins-100)
    
    fig = plt.figure()
    # set height ratios for subplots
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[2,1]) 
    gs.update(left=0, right=1, wspace=0.0, hspace=0.0)

    # x 
    ax0 = plt.subplot(gs[0,0])
    if Comp_Trajs is not None:
        ax0.hist(Comp_Trajs[:,0], bins=bins, density=True, color=c2, histtype='step', linewidth=3)
        ax0.hist(Comp_Trajs[:,0], bins=bins, density=True, color=c2, linewidth=3, alpha=alph)
    if evaluate_weights is True:
        ax0.hist(Trajs_Baseline[:,0], bins=bins, density=True, color=c3, histtype='step', linewidth=3)
        ax0.hist(Trajs_Baseline[:,0], bins=bins, density=True, color=c3, linewidth=3, alpha=alph)
    ax0.hist(Trajs[:,0], weights=weights, bins=bins, density=True, color=c1, histtype='step', linewidth=3)
    ax0.hist(Trajs[:,0], weights=weights, bins=bins, density=True, color=c1, linewidth=3, alpha=alph)
    ax0.set_ylabel('$P$')
    ax0.yaxis.label.set_color(c1)
    ax0.tick_params(axis='y', colors=c1)
    

    # x y
    ax1 = plt.subplot(gs[1,0])
    if space == 'Config':
        ax1.hist2d(Trajs[:,0], Trajs[:,1], weights=weights, bins=n_bins, range=[[-2,2], [-2,2]], cmap='jet')
    elif space == 'Latent':
        ax1.hist2d(Trajs[:,0], Trajs[:,1], weights=weights, bins=n_bins-100, range=[[-4,4], [-4,4]], cmap='jet')
    ax1.set_xlabel('$x$',fontsize=40)
    ax1.set_ylabel('$y$',fontsize=40)

    # y the
    ax2 = plt.subplot(gs[1,1])
    if Comp_Trajs is not None:
        ax2.hist(Comp_Trajs[:,1], bins=bins, density=True, color=c2, histtype='step', linewidth=3, orientation='horizontal')
        ax2.hist(Comp_Trajs[:,1], bins=bins, density=True, color=c2, linewidth=3, alpha=alph, orientation='horizontal')
    if evaluate_weights is True:
        ax2.hist(Trajs_Baseline[:,1], bins=bins, density=True, color=c3, histtype='step', linewidth=3, orientation='horizontal')
        ax2.hist(Trajs_Baseline[:,1], bins=bins, density=True, color=c3, linewidth=3, alpha=alph, orientation='horizontal')
    ax2.hist(Trajs[:,1], weights=weights, bins=bins, density=True, color=c1, histtype='step', linewidth=3, orientation='horizontal')
    ax2.hist(Trajs[:,1], weights=weights, bins=bins, density=True, color=c1, linewidth=3, alpha=alph, orientation='horizontal')
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_xlabel('$P$')
    ax2.xaxis.label.set_color(c1)
    ax2.tick_params(axis='x', colors=c1)

    plt.show();

def Plot_Hist_Alt(Trajs, Comp_Trajs=None, space='Config', n_bins=500, reactive=False, weights=None,
              evaluate_weights=False, weight_threshold_floor=0, weight_threshold_ceiling=30):
    '''
    Plot a histogram in x,y
    '''
    # Params
    plt.rcParams['figure.figsize'] = 10, 13.3
    c1 = 'blue'
    c2 = 'red'
    c3 = 'teal'
    alph = 0.21 
   
    if space == 'Config':
        bins = np.linspace(-2,2, n_bins)
    elif space == 'Latent':
        bins = np.linspace(-4,4, n_bins-100)
    
    fig = plt.figure()
    # set height ratios for subplots
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[2,1]) 
    gs.update(left=0, right=1, wspace=0.0, hspace=0.0)

    Trajs_x = Trajs[:,:,:,0].flatten()
    Trajs_y = Trajs[:,:,:,1].flatten()
    if weights is not None:
        weights_x = weights[:,:,:,0].flatten()
        weights_y = weights[:,:,:,1].flatten()
    else:
        weights_x = None
        weights_y = None

    # x 
    ax0 = plt.subplot(gs[0,0])
    if Comp_Trajs is not None:
        Comp_Trajs_x = Comp_Trajs[:,:,:,0].flatten()
        Comp_Trajs_y = Comp_Trajs[:,:,:,1].flatten()
        ax0.hist(Comp_Trajs_x, bins=bins, density=True, color=c2, histtype='step', linewidth=3)
        ax0.hist(Comp_Trajs_x, bins=bins, density=True, color=c2, linewidth=3, alpha=alph)
    if evaluate_weights is True:
        ax0.hist(Trajs_x, bins=bins, density=True, color=c3, histtype='step', linewidth=3)
        ax0.hist(Trajs_x, bins=bins, density=True, color=c3, linewidth=3, alpha=alph)
    ax0.hist(Trajs_x, weights=weights_x, bins=bins, density=True, color=c1, histtype='step', linewidth=3)
    ax0.hist(Trajs_x, weights=weights_y, bins=bins, density=True, color=c1, linewidth=3, alpha=alph)
    ax0.set_ylabel('$P$')
    ax0.yaxis.label.set_color(c1)
    ax0.tick_params(axis='y', colors=c1)
    

    # x y
    ax1 = plt.subplot(gs[1,0])
    if space == 'Config':
        ax1.hist2d(Trajs_x, Trajs_y, weights=weights_x, bins=n_bins, range=[[-2,2], [-2,2]], cmap='jet')
    elif space == 'Latent':
        ax1.hist2d(Trajs_x, Trajs_y, weights=weights_x, bins=n_bins-100, range=[[-4,4], [-4,4]], cmap='jet')
    ax1.set_xlabel('$x$',fontsize=40)
    ax1.set_ylabel('$y$',fontsize=40)

    # y the
    ax2 = plt.subplot(gs[1,1])
    if Comp_Trajs is not None:
        ax2.hist(Comp_Trajs_y, bins=bins, density=True, color=c2, histtype='step', linewidth=3, orientation='horizontal')
        ax2.hist(Comp_Trajs_y, bins=bins, density=True, color=c2, linewidth=3, alpha=alph, orientation='horizontal')
    if evaluate_weights is True:
        ax2.hist(Trajs_y, bins=bins, density=True, color=c3, histtype='step', linewidth=3, orientation='horizontal')
        ax2.hist(Trajs_y, bins=bins, density=True, color=c3, linewidth=3, alpha=alph, orientation='horizontal')
    ax2.hist(Trajs_y, weights=weights_y, bins=bins, density=True, color=c1, histtype='step', linewidth=3, orientation='horizontal')
    ax2.hist(Trajs_y, weights=weights_y, bins=bins, density=True, color=c1, linewidth=3, alpha=alph, orientation='horizontal')
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_xlabel('$P$')
    ax2.xaxis.label.set_color(c1)
    ax2.tick_params(axis='x', colors=c1)

    plt.show();


def Plot_Angles(Trajs, Comp_Trajs=None, space='Config', show_path=False, n_bins=500, reactive=False, weights=None,
                weight_threshold_floor=0, weight_threshold_ceiling=30, model=None):
    '''
    Plot histogram in x,y and show average and ssd of theta wrt x and y
    '''
    # Params
    plt.rcParams['figure.figsize'] = 10, 13.3
    c1 = 'blue'
    c2 = 'red'
    alph = 0.3

    Trajs[:,:,2] = PM_pi(Trajs[:,:,2])

    if reactive:
        if weights is None:
            Trajs = get_reac(Trajs, potential=model)
        else:
            Trajs, weights = get_reac(Trajs, potential=model, weights=weights)
            # remove unrealistic trajs
            realistic_traj_indicies = np.where(np.logical_and(weights>weight_threshold_floor, weights<weight_threshold_ceiling))[0]
            weights = weights[realistic_traj_indicies]
            Trajs = Trajs[realistic_traj_indicies]
    if show_path:
        chosen_path = Trajs[np.random.randint(len(Trajs))]

    if weights is not None:
        # remove unrealistic trajs
        realistic_traj_indicies = np.where(np.logical_and(weights>weight_threshold_floor, weights<weight_threshold_ceiling))[0]
        weights = weights[realistic_traj_indicies]
        Trajs = Trajs[realistic_traj_indicies]
        weights = np.tile(weights, reps=(np.shape(Trajs)[1]-1,1)).T
        weights = np.concatenate(weights)        
    Trajs = np.concatenate(Trajs) # or we get peak at start
    Trajs[:,2] = PM_pi(Trajs[:,2])
    
    if Comp_Trajs is not None:
        if reactive:
            Comp_Trajs = get_reac(Comp_Trajs, potential=model)
        Comp_Trajs = np.concatenate(Comp_Trajs[:,1:]) # or we get peak at start
        Comp_Trajs[:,2] = np.where(Comp_Trajs[:,2]<=math.pi, Comp_Trajs[:,2], Comp_Trajs[:,2]% -math.pi)

    if space == 'Config':
        bins = np.linspace(-2,2, n_bins)
    elif space == 'Latent':
        bins = np.linspace(-4,4, n_bins)
    
    # x,the Avg & STD
    x_indices = np.digitize(Trajs[:,0], bins=bins)
    xt_avgs = []
    xt_stds = []
    x_bins = bins
    num_invalid = 0
    for i, n in enumerate(bins):
        x_bin = Trajs[:,2][x_indices==i]
        if weights is not None:
            x_weights = weights[x_indices==i]
            if np.sum(x_weights) == 0:
                x_bins = np.delete(x_bins, i - num_invalid)
                num_invalid += 1
                continue 
        else:
            x_weights = None
        xt_avg = np.average(x_bin, weights=x_weights)
        tiled_means = np.tile(xt_avg, reps=len(x_bin))
        xt_var = np.average((x_bin - tiled_means)**2, axis=0, weights=x_weights)
        xt_std = np.sqrt(xt_var)
        xt_avgs.append(xt_avg)
        xt_stds.append(xt_std)
    xt_avgs = np.array(xt_avgs)
    xt_stds = np.array(xt_stds)

    if Comp_Trajs is not None:
        Comp_x_indices = np.digitize(Comp_Trajs[:,0], bins=bins)
        Comp_xt_avgs = []
        Comp_xt_stds = []
        for i, n in enumerate(bins):
            Comp_x_bin = Comp_Trajs[:,2][Comp_x_indices==i]
            Comp_xt_avg = np.average(Comp_x_bin)
            Comp_tiled_means = np.tile(Comp_xt_avg, reps=len(Comp_x_bin))
            Comp_xt_var = np.average((Comp_x_bin - Comp_tiled_means)**2, axis=0)
            Comp_xt_std = np.sqrt(Comp_xt_var)
            Comp_xt_avgs.append(Comp_xt_avg)
            Comp_xt_stds.append(Comp_xt_std)    
        Comp_xt_avgs = np.array(Comp_xt_avgs)
        Comp_xt_stds = np.array(Comp_xt_stds)

    # y,the Avg & STD
    y_indices = np.digitize(Trajs[:,1], bins=bins)
    yt_avgs = []
    yt_stds = []
    y_bins = bins
    num_invalid = 0
    for i, n in enumerate(bins):
        y_bin = Trajs[:,2][y_indices==i]
        if weights is not None:
            y_weights = weights[y_indices==i]
            if np.sum(y_weights) == 0:
                y_bins = np.delete(y_bins, i - num_invalid)
                num_invalid += 1
                continue
        else:
            y_weights = None
        yt_avg = np.average(y_bin, weights=y_weights)
        tiled_means = np.tile(yt_avg, reps=len(y_bin))
        yt_var = np.average((y_bin - tiled_means)**2, axis=0, weights=y_weights)
        yt_std = np.sqrt(yt_var)
        yt_avgs.append(yt_avg)
        yt_stds.append(yt_std)
    yt_avgs = np.array(yt_avgs)
    yt_stds = np.array(yt_stds)

    if Comp_Trajs is not None:
        Comp_y_indices = np.digitize(Comp_Trajs[:,1], bins=bins)
        Comp_yt_avgs = []
        Comp_yt_stds = []
        for i, n in enumerate(bins):
            Comp_y_bin = Comp_Trajs[:,2][Comp_y_indices==i]
            Comp_yt_avg = np.average(Comp_y_bin)
            Comp_tiled_means = np.tile(Comp_yt_avg, reps=len(Comp_y_bin))
            Comp_yt_var = np.average((Comp_y_bin - Comp_tiled_means)**2, axis=0)
            Comp_yt_std = np.sqrt(Comp_yt_var)
            Comp_yt_avgs.append(Comp_yt_avg)
            Comp_yt_stds.append(Comp_yt_std)    
        Comp_yt_avgs = np.array(Comp_yt_avgs)
        Comp_yt_stds = np.array(Comp_yt_stds)
    
    fig = plt.figure()
    # set height ratios for subplots
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[2,1]) 
    gs.update(left=0, right=1, wspace=0.0, hspace=0.0)

    # x the
    ax0 = plt.subplot(gs[0,0])
    if Comp_Trajs is not None:
        ax0.plot(bins, Comp_xt_avgs, color=c2, linewidth=3)
        ax0.fill_between(bins, Comp_xt_avgs+Comp_xt_stds, Comp_xt_avgs-Comp_xt_stds, alpha=alph, color=c2)
    ax0.plot(x_bins, xt_avgs, color=c1, linewidth=3)
    ax0.fill_between(x_bins, xt_avgs+xt_stds, xt_avgs-xt_stds, alpha=alph, color=c1)
    ax0.set_ylim(-math.pi, math.pi)
    ax0.set_xlim(-2, 2)
    ax0.set_ylabel(r'$\theta\pm\sigma_{\theta}$')
    ax0.set_yticks(ticks=[-math.pi/2, 0, math.pi/2, math.pi])
    ax0.set_yticklabels(labels=['$-\pi /2$', '0', '$\pi/2$', '$\pi$'])
    ax0.yaxis.label.set_color(c1)
    ax0.tick_params(axis='y', colors=c1)

    # x y
    ax1 = plt.subplot(gs[1,0])
    if space == 'Config':
        ax1.hist2d(Trajs[:,0], Trajs[:,1], n_bins, weights=weights, range=[[-2,2], [-2,2]], cmap='jet')
    elif space == 'Latent':
        ax1.hist2d(Trajs[:,0], Trajs[:,1], n_bins-100, weights=weights, range=[[-4,4], [-4,4]], cmap='jet')

    if show_path:
        ax1.plot(chosen_path[:,0], chosen_path[:,1], color='white', linewidth=3)
    
    ax1.set_xlabel('$x$',fontsize=40)
    ax1.set_ylabel('$y$',fontsize=40)

    # y the
    ax2 = plt.subplot(gs[1,1])
    if Comp_Trajs is not None:
        ax2.plot(Comp_yt_avgs, bins, color=c2, linewidth=3)
        ax2.fill_betweenx(bins, Comp_yt_avgs+Comp_yt_stds, Comp_yt_avgs-Comp_yt_stds, alpha=alph, color=c2)
    ax2.plot(yt_avgs, y_bins, color=c1, linewidth=3)
    ax2.fill_betweenx(y_bins, yt_avgs+yt_stds, yt_avgs-yt_stds, alpha=alph, color=c1)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_xlim(-math.pi, math.pi)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel(r'$\theta\pm\sigma_{\theta}$')
    ax2.set_xticks(ticks=[-math.pi/2, 0, math.pi/2, math.pi])
    ax2.set_xticklabels(labels=['$-\pi /2$', '0', '$\pi/2$', '$\pi$'])
    ax2.xaxis.label.set_color(c1)
    ax2.tick_params(axis='x', colors=c1)

    plt.show();
    
def get_times(Trajs, potential, dt, weights=None):
    '''
    Return an array containing only valid paths, and the lenghts of each of these paths
    '''
    Paths = []
    Path_times = []

    if weights is not None:
        Path_weights = []

    for i, traj in enumerate(Trajs):
        right_mask = traj[:,0] > 0

        if np.any(right_mask):
            energies = potential.energy(traj)
            energy_mask = energies < 2

            r_e_mask = np.logical_and(energy_mask,right_mask)

            if np.any(r_e_mask):    # if we have values in prod basin
                then_right = np.where(r_e_mask)[0]
                if np.any(then_right):     # if we visit the right well after the left well
                    end = then_right.min() 

                    path = traj[:end+1]     # this is the reactive path!
                    Paths.append(path)

                    time = end
                    Path_times.append((time - 1) * dt)
                    if weights is not None:
                        Path_weights.append(weights[i])

    if weights is None:
        return Paths, np.array(Path_times)
    else:
        return Paths, np.array(Path_times), Path_weights

def Plot_Paths(Trajs, potential, weights=None, weight_threshold_floor=0, weight_threshold_ceiling=30):
    '''
    Plot the shortest, modal lenght, and longest paths
    '''
    # Params
    plt.rcParams['figure.figsize'] = 16, 5
    plt.rcParams.update({'font.size': 20})
    lw = 2

    # Add the starting point
    if weights is None:
        Paths, Path_times = get_times(Trajs, potential)
    else:
        realistic_traj_indicies = np.where(np.logical_and(weights>weight_threshold_floor, weights<weight_threshold_ceiling))[0]
        weights = weights[realistic_traj_indicies]
        Trajs = Trajs[realistic_traj_indicies]
        Paths, Path_times, Path_weights = get_times(Trajs, potential, weights)

    shortest_index = np.argmin(Path_times)
    shortest_lenght = Path_times[shortest_index]
    modal_lenght = stats.mode(np.array(Path_times))[0][0]
    modal_index = np.random.choice(np.where(Path_times == modal_lenght)[0])
    longest_index = np.argmax(Path_times)
    longest_lenght = Path_times[longest_index]

    shortest_path = Paths[shortest_index]
    modal_path = Paths[modal_index]
    longest_path = Paths[longest_index]

    x_1s = np.linspace(-2,2, 100)
    X1grid, X2grid = np.meshgrid(x_1s, x_1s) # x_2s = x_1s
    Xs = np.vstack([X1grid.flatten(), X2grid.flatten()]).T
    energies = potential.energy(Xs)
    energies = energies.reshape((100,100))
    energies = np.minimum(energies, 26.0)          #sets maximum possible energy to 10.0 (replaces any values >10 with 10)

    fig, axs = plt.subplots(1, 3, sharey=True)

    axs[0].plot(shortest_path[:,0], shortest_path[:,1], color='white', linewidth=lw)
    axs[0].contourf(X1grid, X2grid, energies, 50, cmap='jet', vmax=26)
    axs[0].set_xlabel('$x$', fontsize=25)
    axs[0].set_ylabel('$y$', fontsize=25)
    axs[0].set_xlim(-2,2)
    axs[0].set_ylim(-2,2)
    axs[0].text(0.97, 0.97, 't = '+"{:.2f}".format(shortest_lenght), horizontalalignment='right', verticalalignment='top', transform=axs[0].transAxes, 
                color='white', fontweight='bold')

    axs[1].plot(modal_path[:,0], modal_path[:,1], color='white', linewidth=lw)
    axs[1].contourf(X1grid, X2grid, energies, 50, cmap='jet', vmax=26)
    axs[1].set_xlabel('$x$', fontsize=25)
    axs[1].set_xlim(-2,2)
    axs[1].set_ylim(-2,2)
    axs[1].text(0.97, 0.97, 't = '+"{:.2f}".format(modal_lenght), horizontalalignment='right', verticalalignment='top', transform=axs[1].transAxes, 
                color='white', fontweight='bold')

    axs[2].plot(longest_path[:,0], longest_path[:,1], color='white', linewidth=lw)
    axs[2].contourf(X1grid, X2grid, energies, 50, cmap='jet', vmax=26)
    axs[2].set_xlabel('$x$', fontsize=25)
    axs[2].set_xlim(-2,2)
    axs[2].set_ylim(-2,2)
    axs[2].text(0.97, 0.97, 't = '+"{:.2f}".format(longest_lenght), horizontalalignment='right', verticalalignment='top', transform=axs[2].transAxes, 
                color='white', fontweight='bold')

    fig.tight_layout()

    if weights is None:
        print('Percentage Valid:', str(100*len(Paths)/len(Trajs)), '%')
    else:
        print('Percentage Valid:', str(100*np.sum(Path_weights)/np.sum(weights)), '%')

def Plot_Times_Hist(Trajs, potential, color, label, dt, weights=None, end=False, bars=32, fsize=(17, 10), t_max=6.2,
                    weight_threshold_floor=0, weight_threshold_ceiling=30, append=False):
    '''
    Plot a histogram of lenghts 
    '''
    # Params
    alph = 0.35
    plt.rcParams['figure.figsize'] = fsize
    fs = 40

    # Generate Lenghts
    if weights is None:
        _, Path_times = get_times(Trajs, potential, dt=dt)
    else:
        realistic_traj_indicies = np.where(np.logical_and(weights>weight_threshold_floor, weights<weight_threshold_ceiling))[0]
        weights = weights[realistic_traj_indicies]
        Trajs = Trajs[realistic_traj_indicies]
        _, Path_times, weights = get_times(Trajs, potential, weights=weights, dt=dt)

    if append:
        Path_times += dt

    plt.hist(Path_times, bars, weights=weights, density=True, color=color, range=(0, t_max), label=label, histtype='step', linewidth=3)
    plt.hist(Path_times, bars, weights=weights, density=True, color=color, range=(0, t_max), alpha=0.2)
    plt.xlim(0,t_max)
    plt.xlabel(r'$t/ \tau_{\theta}$', fontsize=fs)
    plt.ylabel(r'$P_{TPT}(t)$', fontsize=fs)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))

    if end == True:
        plt.legend(fontsize=32)
        plt.show()
        
def STD(Trajs, Gen_Trajs=None, append=True):

    x_plot_lims = -math.pi
    y_plot_lims = -math.pi

    if append == True:
        starts = np.zeros(shape=(len(Trajs),1,1))
        Trajs = np.concatenate([starts, PM_pi(Trajs)], axis=1)
    else: 
        Trajs = PM_pi(Trajs)

    means = np.average(Trajs, axis=0)
    tiled_means = np.tile(means, reps=(len(Trajs),1,1))
    var = np.average((Trajs - tiled_means)**2, axis=0)
    stds = np.sqrt(var)
    
    p_std = (means + stds)[:,0]
    n_std = (means - stds)[:,0]
    ts = np.arange(np.shape(Trajs)[1]) * dt

    random_traj = Trajs[random.randint(0, len(Trajs))]

    plt.figure(figsize=(10,8))
    if Gen_Trajs is None: 
        # Plot for X
        plt.fill_between(ts, p_std, n_std, color = 'grey', alpha = 0.5)
        plt.plot(ts, random_traj, color='red', linewidth=2)
        plt.xlim(0,3.2)
        plt.ylim(x_plot_lims)
        plt.xlabel('$t$')
        plt.ylabel(r'$\theta$')

    else:
        if append == True:       
            Gen_Trajs = np.concatenate([starts, PM_pi(Gen_Trajs)], axis=1)
        gen_means = np.average(Gen_Trajs, axis=0)
        gen_tiled_means = np.tile(gen_means, reps=(len(Gen_Trajs),1,1))
        gen_var = np.average((Gen_Trajs - gen_tiled_means)**2, axis=0)
        gen_stds = np.sqrt(gen_var)
        
        gen_p_std = (gen_means + gen_stds)[:,0]
        gen_n_std = (gen_means - gen_stds)[:,0]

        gen_random_traj = Gen_Trajs[random.randint(0, len(Gen_Trajs))]

        # Plot for X
        plt.fill_between(ts, gen_p_std, gen_n_std, color = 'blue', alpha = 0.5)    # One std from mean for Generated
        plt.plot(ts, gen_random_traj, color='blue', linewidth=2)                        # Random generated trajectory
        plt.plot(ts, p_std, color='black', linewidth=2, linestyle='--')             # Target std+mean 
        plt.plot(ts, n_std, color='black', linewidth=2, linestyle='--')
        plt.plot(ts, random_traj, color='black')                                        # Random target trajectory
        plt.xlim(0,3.2)
        plt.ylim(x_plot_lims)
        plt.xlabel('$t$')
        plt.ylabel(r'$\theta$')
    
    plt.show()



def Plt_Dist(Dist, Comp_Dist=None, n_bins=100):
    '''
    Plot histogram of generated latent distribution
    '''
    c1 = 'blue'
    c2 = 'red'
    alph = 0.2

    # Make bins
    bins = np.linspace(-0.5 -math.pi, math.pi +0.5, n_bins)

    # x noise
    plt.figure(figsize=(10,8))
    plt.axvline(x=-math.pi, color='black')
    plt.axvline(x=math.pi, color='black')
    plt.hist(Dist[:,0:].flatten(), bins, color=c1, density=True, alpha=alph)
    if Comp_Dist is not None:
        plt.hist(Comp_Dist[:,0:].flatten(), bins, color=c2, density=True, histtype='step')
    plt.xlim(-0.5 -math.pi, math.pi +0.5)
    plt.ylabel('$P$', color=c1)
    plt.xlabel(r'$T^{-1}(\theta)$')
    plt.tick_params(axis='x', colors=c1)
    plt.show()


def Recover_Angle_Noise(Angs, rot_coeff, n_bins=250):
    '''
    Plot histograms of the noise of angle increments.
    '''
    # Params
    c1 = 'blue'
    alph = 0.2
    ablim = 3 # absolute limit

    ### Recover noise ###
    Ang_noise = PM_pi(np.diff(Angs, axis=1))
    Ang_noise = np.concatenate(Ang_noise)

    ## Target
    VonMisesVector = np.random.vonmises(0, 1/(rot_coeff**2), size=(np.shape(Angs)))

    # Make bins
    bins = np.linspace(-ablim, ablim, n_bins)

    # x noise
    plt.figure(figsize=(10,8))
    plt.hist(Ang_noise, bins, color=c1, density=True, alpha=alph)
    plt.hist(VonMisesVector.flatten(), bins, color='red', density=True, histtype='step')
    plt.ylabel('$P$', color=c1)
    plt.tick_params(axis='y', colors=c1)

    plt.show();
