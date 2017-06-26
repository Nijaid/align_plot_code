import numpy as np
import pylab as plt
from jlu.microlens import residuals
from jlu.microlens import align_compare
from jlu.microlens import trim_starlists
from jlu.microlens import align_epochs
# from jlu.microlens import model
#from jlu.util import fileUtil
from astropy.table import Table
import numpy as np
import os
import shutil
# from gcwork import starTables
from gcwork import starset
from gcwork import objects
from scipy import spatial
import scipy
import scipy.stats
from time import strftime, localtime
import pdb


def get_align(align_dir="./", align_root='/align'):
    s = starset.StarSet(align_dir + align_root)

    name = s.getArray('name')

    x = s.getArrayFromAllEpochs('xpix')
    y = s.getArrayFromAllEpochs('ypix')
    xe_p = s.getArrayFromAllEpochs('xpixerr_p')
    ye_p = s.getArrayFromAllEpochs('ypixerr_p')
    xe_a = s.getArrayFromAllEpochs('xpixerr_a')
    ye_a = s.getArrayFromAllEpochs('ypixerr_a')

    time = s.years

    x0 = s.getArray('fitpXv.p')
    vx = s.getArray('fitpXv.v')
    t0x = s.getArray('fitpXv.t0')
    y0 = s.getArray('fitpYv.p')
    vy = s.getArray('fitpYv.v')
    t0y = s.getArray('fitpYv.t0')

    N_epochs = len(s.years)
    N_stars = len(s.stars)

    data = {'x': x, 'y': y, 'xpixerr_p': xe_p, 'ypixerr_p': ye_p,
            'xpixerr_a': xe_a, 'ypixerr_a': ye_a,
            'N_epochs': N_epochs, 'N_stars': N_stars, 'years': time}
    return data

def raw_align_plot(targets, align_dir="./"):
    plot_dir = '/u/nijaid/microlens/align_plots/' + targets[0]
    if os.path.exists(plot_dir) == False:
        os.mkdir(plot_dir)

    s = starset.StarSet(align_dir + '/align')

    name = s.getArray('name')

    x = s.getArrayFromAllEpochs('xpix')
    y = s.getArrayFromAllEpochs('ypix')
    xe_p = s.getArrayFromAllEpochs('xpixerr_p')
    ye_p = s.getArrayFromAllEpochs('ypixerr_p')
    xe_a = s.getArrayFromAllEpochs('xpixerr_a')
    ye_a = s.getArrayFromAllEpochs('ypixerr_a')

    N_epochs = len(s.years)
    N_stars = len(s.stars)

    dx = np.zeros((N_epochs, N_stars), dtype=float)
    dy = np.zeros((N_epochs, N_stars), dtype=float)
    dxp = np.zeros((N_epochs, N_stars), dtype=float)
    dyp = np.zeros((N_epochs, N_stars), dtype=float)
    dxa = np.zeros((N_epochs, N_stars), dtype=float)
    dya = np.zeros((N_epochs, N_stars), dtype=float)

    for ff in range(N_epochs):
        # Clean out undetected stars
        idx = np.where((xe_p[ff, :] != 0) & (ye_p[ff, :] != 0))[0]

        # Put together observed data
        dx[ff, idx] = x[ff, idx]
        dy[ff, idx] = y[ff, idx]
        dxp[ff, idx] = xe_p[ff, idx]
        dyp[ff, idx] = ye_p[ff, idx]
        dxa[ff, idx] = xe_a[ff, idx]
        dya[ff, idx] = ye_a[ff, idx]

    tdx = [name.index(targets[0]), name.index(targets[1]), name.index(targets[2])]

    # Position plot
    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.plot(s.years, dx[:, tdx[0]], color='red', linestyle='none', marker='.', label=targets[0])
    plt.plot(s.years, dx[:, tdx[1]], color='blue', linestyle='none', marker='.', label=targets[1])
    plt.plot(s.years, dx[:, tdx[2]], color='green', linestyle='none', marker='.', label=targets[2])
    plt.tick_params(labelsize=6)
    plt.ylabel('x (pix)')
    plt.title('position')
    plt.legend(numpoints=2, fontsize=8)

    plt.subplot(212)
    plt.plot(s.years, dy[:, tdx[0]], color='red', linestyle='none', marker='.', label=targets[0])
    plt.plot(s.years, dy[:, tdx[1]], color='blue', linestyle='none', marker='.', label=targets[1])
    plt.plot(s.years, dy[:, tdx[2]], color='green', linestyle='none', marker='.', label=targets[2])
    plt.tick_params(labelsize=6)
    plt.ylabel('y (pix)')
    plt.xlabel('Year')

    plt.tight_layout()
    plt.savefig(plot_dir + '/' + targets[0] + '_posplot.png')

    # Error plot
    plt.figure(2)
    plt.clf()
    plt.subplot(221) # x photo error plot
    plt.plot(s.years, dxp[:, tdx[0]], color='red', linestyle='none', marker='.', label=targets[0])
    plt.plot(s.years, dxp[:, tdx[1]], color='blue', linestyle='none', marker='.', label=targets[1])
    plt.plot(s.years, dxp[:, tdx[2]], color='green', linestyle='none', marker='.', label=targets[2])
    plt.tick_params(labelsize=6)
    plt.ylabel('x (pix)')
    plt.title('error in position')
    plt.axhline(0, color='k', linestyle='--')

    plt.subplot(223) # y photo error plot
    plt.plot(s.years, dyp[:, tdx[0]], color='red', linestyle='none', marker='.', label=targets[0])
    plt.plot(s.years, dyp[:, tdx[1]], color='blue', linestyle='none', marker='.', label=targets[1])
    plt.plot(s.years, dyp[:, tdx[2]], color='green', linestyle='none', marker='.', label=targets[2])
    plt.tick_params(labelsize=6)
    plt.ylabel('y (pix)')
    plt.xlabel('Year')
    plt.axhline(0, color='k', linestyle='--')

    plt.subplot(222) # x astrometric error plot
    plt.plot(s.years, dxa[:, tdx[0]], color='red', linestyle='none', marker='.', label=targets[0])
    plt.plot(s.years, dxa[:, tdx[1]], color='blue', linestyle='none', marker='.', label=targets[1])
    plt.plot(s.years, dxa[:, tdx[2]], color='green', linestyle='none', marker='.', label=targets[2])
    plt.tick_params(labelsize=6)
    plt.legend(numpoints=2, fontsize=8)
    plt.title('error in alignment')
    plt.axhline(0, color='k', linestyle='--')

    plt.subplot(224) # y astrometric error plot
    plt.plot(s.years, dya[:, tdx[0]], color='red', linestyle='none', marker='.', label=targets[0])
    plt.plot(s.years, dya[:, tdx[1]], color='blue', linestyle='none', marker='.', label=targets[1])
    plt.plot(s.years, dya[:, tdx[2]], color='green', linestyle='none', marker='.', label=targets[2])
    plt.tick_params(labelsize=6)
    plt.xlabel('Year')
    plt.axhline(0, color='k', linestyle='--')

    plt.tight_layout()
    out_ex = plot_dir + '/' + targets[0] + '_errplot.png'
    plt.savefig(out_ex)

    print('Saved ' + out_ex)


date = strftime('%Y_%m_%d', localtime())

def var_align(target, stars, epochs, refEpoch, date=date,
            transforms=[3,4,5], magCuts=[22], weights=[1,2,3,4],
            trimStars=True, restrict=True):
    """
    Creates the lists necessary for the alignment loop, and runs the loop itself.
    """
    work_dir='a_'+date
    root_dir = '/u/nijaid/work/' + target.upper() + '/'
    template_dir = root_dir + work_dir + '/a_' + target + '_' + date
    if template_dir[len(template_dir)-1] != '/':
        template_dir = template_dir + '/'
    
    if trimStars==True: # Trim starlists to a radius of 8"
        trim_starlists.trim_in_radius(Readpath=template_dir+'lis/',
                        TargetName=target, epochs=epochs, radius_cut_in_mas=4500.0)

    # make the align.lis
    align_epochs.make_align_list(root=root_dir, prefix = 'a', date=date,
                                 target=target, refEpoch=refEpoch)
    print(template_dir)

    # run the alignment loop and plot
    align_epochs.align_loop(root=root_dir, prefix='a', target=target, stars=stars, date=date,
            transforms=transforms, magCuts=magCuts, weightings=weights,
            Nepochs=str(len(epochs)), overwrite=True, nMC=100,
            makePlots=True, DoAlign=True, restrict=restrict)


def plot_20stars(work_dir="./"):
    dirs = os.listdir(work_dir)
    _dirs = []
    for dd in dirs:
        if len(dd) == 37:
            _dirs.append(dd)
            
    for ii in range(len(_dirs)):
        _dir = work_dir + '/' +  _dirs[ii] + '/'
        align_dir = _dir + 'align/'
        _f = Table.read(align_dir + 'align_t.name', format='ascii')
        names = _f['col1']
        names = np.array(names)
        pdb.set_trace()
        for s in range(20):
            plotStar(starName=names[s], rootDir=_dir, align='align/align_t', poly='polyfit_d/fit', points='/points_d/')
    
def plotStar(starName,rootDir='./', align='align/align_t', poly='polyfit_d/fit',
                 points='/points_d/', radial=False, NcolMax=1, figsize=(15,10)):
    print('Plotting residuals for ' + starName + 'in ' + rootDir)
    Nrows = 3

    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)
    py.close('all')
    py.figure(2, figsize=figsize)
    names = s.getArray('name')
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x,y)

    ind = names.index(starName)
    star = s.stars[ind]

    


def align_plot_fit(targets, align_dir="./"):
    plot_dir = '/u/nijaid/microlens/align_plots/' + targets[0]
    if os.path.exists(plot_dir) == False:
        os.mkdir(plot_dir)

    s = starset.StarSet(align_dir + '/align')

    name = s.getArray('name')

    x = s.getArrayFromAllEpochs('xpix')
    y = s.getArrayFromAllEpochs('ypix')
    xe_p = s.getArrayFromAllEpochs('xpixerr_p')
    ye_p = s.getArrayFromAllEpochs('ypixerr_p')
    xe_a = s.getArrayFromAllEpochs('xpixerr_a')
    ye_a = s.getArrayFromAllEpochs('ypixerr_a')

    x0 = s.getArray('fitpXv.p')
    vx = s.getArray('fitpXv.v')
    t0x = s.getArray('fitpXv.t0')
    y0 = s.getArray('fitpYv.p')
    vy = s.getArray('fitpYv.v')
    t0y = s.getArray('fitpYv.t0')

    N_epochs = len(s.years)
    N_stars = len(s.stars)

    dx = np.zeros((N_epochs, N_stars), dtype=float)
    dy = np.zeros((N_epochs, N_stars), dtype=float)
    dxe = np.zeros((N_epochs, N_stars), dtype=float)
    dye = np.zeros((N_epochs, N_stars), dtype=float)

    for ee in range(N_epochs):
        # Identify reference stars that can be used for calculating local distortions.
        # We use all the stars (not restricted to just the alignment stars).
        # Use a KD Tree.
        # First clean out undetected stars.
        idx = np.where((xe_p[ee, :] != 0) & (ye_p[ee, :] != 0))[0]

        # Put together observed data.
        coords = np.empty((len(idx), 2))
        coords[:, 0] = x[ee, idx]
        coords[:, 1] = y[ee, idx]

        tree = spatial.KDTree(coords)

        # For every star, calculate the best fit position at each epoch.
        dt_x = s.years[ee] - t0x   # N_stars long array
        dt_y = s.years[ee] - t0y

        x_fit = x0 + (vx * dt_x)
        y_fit = y0 + (vy * dt_y)

        # Query the KD tree for the nearest 3 neighbors (including self) for every star.
        nn_r, nn_i = tree.query(coords, 3, p=2)

        id1 = idx[nn_i[:, 1]]
        dx1 = x[ee, id1] - x_fit[id1]
        dy1 = y[ee, id1] - y_fit[id1]

        id2 = idx[nn_i[:, 2]]
        dx2 = x[ee, id2] - x_fit[id2]
        dy2 = y[ee, id2] - y_fit[id2]

        dx12 = np.array([dx1, dx2])
        dy12 = np.array([dy1, dy2])

        dx_avg = dx12.mean(axis=0)
        dy_avg = dy12.mean(axis=0)
        dx_std = dx12.std(axis=0)
        dy_std = dy12.std(axis=0)

        # Print the deltas
        lens = np.where(np.array(name) == targets[0])[0][0]
        msg = 'Mean {0:4s} = {1:6.3f} +- {2:6.3f} pix   ({3:6.2f} sigma)'
        print('')
        print('Epoch = {0:d}'.format(ee))
        print(msg.format('dx', dx_avg[lens], dx_std[lens], dx_avg[lens] / dx_std[lens]))
        print(msg.format('dy', dy_avg[lens], dy_std[lens], dy_avg[lens] / dy_std[lens]))

        dx[ee, idx] = dx_avg
        dy[ee, idx] = dy_avg
        dxe[ee, idx] = dx_std
        dye[ee, idx] = dy_std

    # Get the lens and two nearest sources
    tdx = [name.index(targets[0]), name.index(targets[1]), name.index(targets[2])]
    pdb.set_trace()

    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.errorbar(s.years, dx[:, tdx[0]], yerr=dxe[:, tdx[0]], color='red', linestyle='none', marker='.', label=targets[0])
    plt.errorbar(s.years, dx[:, tdx[1]], yerr=dxe[:, tdx[1]], color='blue', linestyle='none', marker='.', label=targets[1])
    plt.errorbar(s.years, dx[:, tdx[2]], yerr=dxe[:, tdx[2]], color='green', linestyle='none', marker='.', label=targets[2])
    plt.legend(numpoints=1, fontsize=8)
    plt.ylim(-0.22, 0.22)
    plt.ylabel(r'$\Delta$x (pix)')
    plt.axhline(0, color='k', linestyle='--')

    plt.subplot(212)
    plt.errorbar(s.years, dy[:, tdx[0]], yerr=dye[:, tdx[0]], color='red', linestyle='none', marker='.', label=targets[0])
    plt.errorbar(s.years, dy[:, tdx[1]], yerr=dye[:, tdx[1]], color='blue', linestyle='none', marker='.', label=targets[1])
    plt.errorbar(s.years, dy[:, tdx[2]], yerr=dye[:, tdx[2]], color='green', linestyle='none', marker='.', label=targets[2])
    plt.ylim(-0.22, 0.22)
    plt.ylabel(r'$\Delta$y (pix)')
    plt.xlabel('Year')
    plt.axhline(0, color='k', linestyle='--')

    plt.savefig(plot_dir + '/plots/plot_local_astrometry.png')
    plt.close()

    return(plot_dir + '/plots/plot_local_astrometry.png')
