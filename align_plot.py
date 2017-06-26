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

plt.close('all')
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
        print('\nAlignment %d of %d' %(ii+1,len(_dirs)))
        _dir = work_dir + '/' +  _dirs[ii] + '/'
        align_dir = _dir + 'align/'
        _f = Table.read(align_dir + 'align_t.name', format='ascii')
        names = _f['col1']
        names = np.array(names)
        for s in range(20):
            plotStar(starName=names[s], rootDir=_dir, align='align/align_t', poly='polyfit_d/fit', points='/points_d/')

def plotStar(starName,rootDir='./', align='align/align_t', poly='polyfit_d/fit', points='/points_d/'):
    print('Plotting residuals for ' + starName + ' in ' + rootDir)

    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)
    names = s.getArray('name')
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x,y)

    plt.close('all')
    plt.figure(1, figsize=(10,15))
    Ncols = 2
    Nrows = 3

    iii = names.index(starName)
    star = s.stars[iii]

    pointsTab = Table.read(rootDir + points + starName + '.points', format='ascii')

    time = pointsTab[pointsTab.colnames[0]]
    x = pointsTab[pointsTab.colnames[1]]
    y = pointsTab[pointsTab.colnames[2]]
    xerr = pointsTab[pointsTab.colnames[3]]
    yerr = pointsTab[pointsTab.colnames[4]]

    fitx = star.fitXv
    fity = star.fitYv
    dt = time - fitx.t0
    fitLineX = fitx.p + (fitx.v * dt)
    fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

    fitLineY = fity.p + (fity.v * dt)
    fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

    diffX = x - fitLineX
    diffY = y - fitLineY
    diff = np.hypot(diffX, diffY)
    rerr = np.sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
    sigX = diffX / xerr
    sigY = diffY / yerr
    sig = diff / rerr


    # Determine if there are points that are more than 5 sigma off
    idxX = np.where(abs(sigX) > 4)
    idxY = np.where(abs(sigY) > 4)
    idx = np.where(abs(sig) > 4)

    print( '\tX Chi^2 = %5.2f (%6.2f for %2d dof)' %
          (fitx.chi2red, fitx.chi2, fitx.dof))
    print( '\tY Chi^2 = %5.2f (%6.2f for %2d dof)' %
          (fity.chi2red, fity.chi2, fity.dof))

    dateTicLoc = plt.MultipleLocator(3)
    dateTicRng = [2006, 2017]
    # dateTics = np.array([2011, 2012, 2013, 2014, 2015, 2016, 2017])
    dateTics = np.array([2015, 2016, 2017])
    DateTicsLabel = dateTics-2000

    # See if we are using MJD instead.
    if time[0] > 50000:
        dateTicLoc = plt.MultipleLocator(1000)
        dateTicRng = [56000, 58000]
        dateTics = np.arange(dateTicRng[0], dateTicRng[-1]+1, 1000)
        DateTicsLabel = dateTics


    maxErr = np.array([xerr, yerr]).max()
    resTicRng = [-1.1*maxErr, 1.1*maxErr]

    from matplotlib.ticker import FormatStrFormatter
    fmtX = FormatStrFormatter('%5i')
    fmtY = FormatStrFormatter('%6.2f')
    fontsize1 = 10

    paxes = plt.subplot(Nrows, Ncols, 1)
    plt.plot(time, fitLineX, 'b-')
    plt.plot(time, fitLineX + fitSigX, 'b--')
    plt.plot(time, fitLineX - fitSigX, 'b--')
    plt.errorbar(time, x, yerr=xerr, fmt='k.')
    rng = plt.axis()
    plt.ylim(np.min(x-xerr-0.1),np.max(x+xerr+0.1))
    plt.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
    if time[0] > 50000:
        plt.xlabel('Date (MJD)', fontsize=fontsize1)
    plt.ylabel('X (pix)', fontsize=fontsize1)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.yaxis.set_major_formatter(fmtY)
    paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
    plt.yticks(np.arange(np.min(x-xerr-0.1), np.max(x+xerr+0.1), 0.2))
    plt.xticks(dateTics, DateTicsLabel)
    plt.xlim(np.min(dateTics), np.max(dateTics))
    plt.annotate(starName,xy=(1.0,1.1), xycoords='axes fraction', fontsize=12, color='red')
    
    paxes = plt.subplot(Nrows, Ncols, 2)
    plt.plot(time, fitLineY, 'b-')
    plt.plot(time, fitLineY + fitSigY, 'b--')
    plt.plot(time, fitLineY - fitSigY, 'b--')
    plt.errorbar(time, y, yerr=yerr, fmt='k.')
    rng = plt.axis()
    plt.axis(dateTicRng + [rng[2], rng[3]], fontsize=fontsize1)
    plt.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
    if time[0] > 50000:
        plt.xlabel('Date (MJD)', fontsize=fontsize1)
    plt.ylabel('Y (pix)', fontsize=fontsize1)
    #paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.yaxis.set_major_formatter(fmtY)
    paxes.tick_params(axis='both', which='major', labelsize=12)
    plt.ylim(np.min(y-yerr-0.1),np.max(y+yerr+0.1))
    plt.yticks(np.arange(np.min(y-yerr-0.1), np.max(y+yerr+0.1), 0.2))
    plt.xticks(dateTics, DateTicsLabel)
    plt.xlim(np.min(dateTics), np.max(dateTics))

    paxes = plt.subplot(Nrows, Ncols, 3)
    plt.plot(time, np.zeros(len(time)), 'b-')
    plt.plot(time, fitSigX, 'b--')
    plt.plot(time, -fitSigX, 'b--')
    plt.errorbar(time, x - fitLineX, yerr=xerr, fmt='k.')
    plt.axis(dateTicRng + resTicRng, fontsize=fontsize1)
    plt.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
    if time[0] > 50000:
        plt.xlabel('Date (MJD)', fontsize=fontsize1)
    plt.ylabel('X Residuals (pix)', fontsize=fontsize1)
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
    plt.xticks(dateTics, DateTicsLabel)
    plt.xlim(np.min(dateTics), np.max(dateTics))

    paxes = plt.subplot(Nrows, Ncols, 4)
    plt.plot(time, np.zeros(len(time)), 'b-')
    plt.plot(time, fitSigY, 'b--')
    plt.plot(time, -fitSigY, 'b--')
    plt.errorbar(time, y - fitLineY, yerr=yerr, fmt='k.')
    plt.axis(dateTicRng + resTicRng, fontsize=fontsize1)
    plt.xlabel('Date -2000 (yrs)', fontsize=fontsize1)
    if time[0] > 50000:
        plt.xlabel('Date (MJD)', fontsize=fontsize1)
    plt.ylabel('Y Residuals (pix)', fontsize=fontsize1)
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
    plt.xticks(dateTics, DateTicsLabel)
    plt.xlim(np.min(dateTics), np.max(dateTics))

    paxes = plt.subplot(Nrows, Ncols, 5)
    plt.errorbar(x,y, xerr=xerr, yerr=yerr, fmt='k.')
    plt.yticks(np.arange(np.min(y-yerr-0.1), np.max(y+yerr+0.1), 0.2))
    plt.xticks(np.arange(np.min(x-xerr-0.1), np.max(x+xerr+0.1), 0.2), rotation = 270)
    plt.axis('equal')
    paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
    paxes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    paxes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('X (pix)', fontsize=fontsize1)
    plt.ylabel('Y (pix)', fontsize=fontsize1)
    plt.plot(fitLineX, fitLineY, 'b-')

    bins = np.arange(-7, 7, 1)
    paxes = plt.subplot(Nrows, Ncols, 6)
    id = np.where(diffY < 0)[0]
    sig[id] = -1.*sig[id]
    (n, b, p) = plt.hist(sigX, bins, histtype='stepfilled', color='b')
    plt.setp(p, 'facecolor', 'b')
    (n, b, p) = plt.hist(sigY, bins, histtype='step', color='r')
    plt.axis([-7, 7, 0, 8], fontsize=10)
    plt.xlabel('X Residuals (sigma)', fontsize=fontsize1)
    plt.ylabel('Number of Epochs', fontsize=fontsize1)

    title = rootDir.split('/')[-2]
    plt.suptitle(title, x=0.5, y=0.97)
    plt.subplots_adjust(wspace=0.4, hspace=0.4, left = 0.15, bottom = 0.1, right=0.9, top=0.9) 
    plt.savefig(rootDir+'plots/plotStar_' + starName + '.png')


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
