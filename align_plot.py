import numpy as np
import pylab as plt
from jlu.microlens import residuals
from jlu.microlens import align_compare
# from jlu.microlens import model
from jlu.util import fileUtil
from astropy.table import Table
import numpy as np
import os
import shutil
# from gcwork import starTables
from gcwork import starset
from scipy import spatial
import scipy
import scipy.stats
import pdb

def get_align(align_dir="./"):
    s = starset.StarSet(align_dir + '/align')

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
            'N_epochs': N_epochs, 'N_stars': N_stars, 'time': time}
    return data

def align_plot(targets, align_dir="./"):
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

    # x0 = s.getArray('fitpXv.p')
    # vx = s.getArray('fitpXv.v')
    # t0x = s.getArray('fitpXv.t0')
    # y0 = s.getArray('fitpYv.p')
    # vy = s.getArray('fitpYv.v')
    # t0y = s.getArray('fitpYv.t0')

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
        pdb.set_trace()

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
    return tdx
    # plt.figure(1)
    # plt.clf()
    # plt.subplot(211)
    # plt.errorbar(s.years, dx[:, tdx[0]], yerr=dxe[:, tdx[0]], color='red', linestyle='none', marker='.', label=targets[0])
    # plt.errorbar(s.years, dx[:, tdx[1]], yerr=dxe[:, tdx[1]], color='blue', linestyle='none', marker='.', label=targets[1])
    # plt.errorbar(s.years, dx[:, tdx[2]], yerr=dxe[:, tdx[2]], color='green', linestyle='none', marker='.', label=targets[2])
    # plt.legend(numpoints=1, fontsize=8)
    # plt.ylim(-0.22, 0.22)
    # plt.ylabel(r'$\Delta$x (pix)')
    # plt.axhline(0, color='k', linestyle='--')
    #
    # plt.subplot(212)
    # plt.errorbar(s.years, dy[:, tdx[0]], yerr=dye[:, tdx[0]], color='red', linestyle='none', marker='.', label=targets[0])
    # plt.errorbar(s.years, dy[:, tdx[1]], yerr=dye[:, tdx[1]], color='blue', linestyle='none', marker='.', label=targets[1])
    # plt.errorbar(s.years, dy[:, tdx[2]], yerr=dye[:, tdx[2]], color='green', linestyle='none', marker='.', label=targets[2])
    # plt.ylim(-0.22, 0.22)
    # plt.ylabel(r'$\Delta$y (pix)')
    # plt.xlabel('Year')
    # plt.axhline(0, color='k', linestyle='--')
    #
    # plt.savefig(plot_dir + '/plots/plot_local_astrometry.png')
    # plt.close()
    #
    # return(plot_dir + '/plots/plot_local_astrometry.png')
