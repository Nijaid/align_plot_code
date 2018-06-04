import numpy as np
import pylab as plt
from microlens.jlu import residuals
from microlens.jlu import align_compare
from microlens.jlu import trim_starlists
from microlens.jlu import align_epochs
# from jlu.microlens import model
#from jlu.util import fileUtil
from astropy.table import Table
import numpy as np
import os
import shutil
from gcwork import starset
from gcwork import objects
from scipy import spatial
from scipy import stats
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


date = strftime('%Y_%m_%d', localtime())

def var_align(target, stars, epochs, refEpoch, date=date, radius_cut=4000.0,
            transforms=[3,4,5], magCuts=[18], weights=[1,2,3,4],
            trimStars=True, restrict=False):
    """
    Creates the lists necessary for the alignment loop, and runs the loop itself.
    """
    work_dir='a_'+date
    root_dir = '/g/lu/microlens/cross_epoch/' + target.upper() + '/'
    template_dir = root_dir + work_dir + '/a_' + target + '_' + date
    if template_dir[len(template_dir)-1] != '/':
        template_dir = template_dir + '/'

    if trimStars==True: # Trim starlists to a radius of 4"
        trim_starlists.trim_in_radius(Readpath=template_dir+'lis/',
                        TargetName=target, epochs=epochs, radius_cut_in_mas=radius_cut)

    # make the align.lis
    align_epochs.make_align_list(root=root_dir, prefix = 'a', date=date,
                                 target=target, refEpoch=refEpoch)
    print(template_dir)

    # run the alignment loop and plot
    align_epochs.align_loop(root=root_dir, prefix='a', target=target, stars=stars, date=date,
            transforms=transforms, magCuts=magCuts, weightings=weights,
            Nepochs=str(len(epochs)), overwrite=True, nMC=100,
            makePlots=True, DoAlign=True, restrict=restrict)


def plot_stars(stars=10, work_dir="./"):
    dirs = os.listdir(work_dir)
    _dirs = []
    for dd in dirs:
        if len(dd) == 37: #hardcoded for structure a_ob######_yyyy_mm_dd_a#_m##_w#_MC###
            _dirs.append(dd)

    for ii in range(len(_dirs)):
        print('\nAlignment %d of %d' %(ii+1,len(_dirs)))
        _dir = work_dir + '/' +  _dirs[ii] + '/'
        align_dir = _dir + 'align/'
        _f = Table.read(align_dir + 'align_t.name', format='ascii')
        names = _f['col1']
        names = np.array(names)
        for s in range(stars):
            plotStar(starName=names[s], rootDir=_dir, align='align/align_t', poly='polyfit_d/fit', points='/points_d/')

    plt.close('all')

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
    dateTicRng = [2012, 2019]
    # dateTics = np.array([2011, 2012, 2013, 2014, 2015, 2016, 2017])
    dateTics = np.array([2015, 2016, 2017, 2018])
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
    (n, b, p) = plt.hist(sigX, bins, histtype='stepfilled', color='b', label='X res')
    plt.setp(p, 'facecolor', 'b')
    (n, b, p) = plt.hist(sigY, bins, histtype='step', color='r', label='Y res')
    plt.axis([-7, 7, 0, 8], fontsize=10)
    plt.xlabel('Residuals (sigma)', fontsize=fontsize1)
    plt.ylabel('Number of Epochs', fontsize=fontsize1)
    plt.legend()

    title = rootDir.split('/')[-2]
    plt.suptitle(title, x=0.5, y=0.97)
    plt.subplots_adjust(wspace=0.4, hspace=0.4, left = 0.15, bottom = 0.1, right=0.9, top=0.9) 
    plt.savefig(rootDir+'plots/plotStar_' + starName + '.png')


def ftest_summary(target, date,root='/u/nijaid/work/', prefix='a',
                orders=[3,4,5], weights=[1,2,3,4], Kcut=18, only_stars_in_fit=True,
                export=False):
    # Make 2D arrays for each alignment parameter
    orders = np.array(orders)
    weights = np.array(weights)

    end = 'all'
    if only_stars_in_fit:
        end = 'used'
        
    work_dir = root + target.upper() + '/' + prefix + '_' + date + '/'

    # Dictionary of free parameters for each order
    N_par_aln_dict = {3: 3, 4: 6, 5: 10}
    N_par_aln = [N_par_aln_dict[order] for order in orders]

    print('********* ' + target.upper() + ' F TEST *********')
    print( '* Results for Kcut={0:d} and in fit={1}'.format(Kcut, only_stars_in_fit) )
    print( '*********' )
    N_free_all = np.zeros((len(weights), len(orders)), dtype=int)
    N_data_all = np.zeros((len(weights), len(orders)), dtype=int)
    ftest = np.zeros((len(weights), len(orders)-1), dtype=float)
    p_value = np.zeros((len(weights), len(orders)-1), dtype=float)
    chi2_all = np.zeros((len(weights), len(orders)), dtype=float)

    for ww in range(len(weights)):
        for oo in range(len(orders)):
            analysis_root_fmt = '{0:s}_{1:s}_{2:s}_a{3:d}_m{4:d}_w{5:d}_MC100/'
            analysis_dir = work_dir + analysis_root_fmt.format(prefix, target, date, orders[oo], Kcut, weights[ww])

            data = residuals.check_alignment_fit(root_dir=analysis_dir)

            year = data['year']
            scale = 9.952 # mas/pixel

            chi2x = data['chi2x_' + end].sum()
            chi2y = data['chi2y_' + end].sum()
            chi2 = chi2x + chi2y

            N_data = 2 * data['N_stars_' + end].sum()
            N_stars_max = data['N_stars_' + end].max()
            N_vfit = 4 * N_stars_max
            N_afit = (len(year) - 1) * N_par_aln[oo] * 2

            N_free_param = N_vfit + N_afit
            N_dof = N_data - N_free_param

            fmt = 'Order={0:d}  Weight={1:d} N_stars = {2:3d}  N_par = {3:3d}  N_dof = {4:3d}  Chi^2 = {5:5.1f}'
            print(fmt.format(orders[oo], weights[ww], int(N_data), int(N_free_param), int(N_dof), chi2))

            N_free_all[ww, oo] = int(N_free_param)
            N_data_all[ww, oo] = int(N_data)
            chi2_all[ww, oo] = chi2

        # F-test
        N_free = N_free_all[ww]
        N_data = N_data_all[ww]
        chi2 = chi2_all[ww]
        f1 = N_data[1:] - N_free[1:]
        f2 = np.diff(N_free)
        ft = (-1 * np.diff(chi2)) / chi2[1:] * f1 / f2

        p = p_value[ww]
        for mm in range(len(ft)):
            p[mm] = stats.f.sf(ft[mm], f2[mm], f1[mm])
            fmt = 'Weight = {0}  O-1 = {1} --> O = {2}   F = {3:5.2f}  p = {4:7.5f}'
            print(fmt.format(weights[ww], orders[mm], orders[mm+1], ft[mm], p[mm]))
        p_value[ww] = p
        ftest[ww] = ft
        print('\n')

def compare_orders(target, date, prefix='a', orders=[3,4,5], weights=[1,2,3,4],
                    Kcut=18, only_stars_in_fit=True, plot_dir='compare_epochs/', root='/g/lu/microlens/cross_epoch/'):
    from microlens.jlu import align_compare
    from nirc2.reduce import util

    work_dir = root + target.upper() + '/' + prefix + '_' + date + '/'
    plot_d = '/u/nijaid/work/' + target.upper() + '/' + prefix + '_' + date + '/' + plot_dir
    util.mkdir(plot_d)
    
    al_fmt = work_dir + prefix + '_' + target + '_' + date + '_a{0}_m{1}_w{2}_MC100/'
    for ww in weights:
        al_dirs = []
        for oo in orders:
            al_dir = al_fmt.format(int(oo), int(Kcut), int(ww))
            al_dirs.append(al_dir)
            
        cut_dir = 'm' + str(Kcut) + '_w' + str(ww) + '/'
        print('** Weight = ' + str(ww) + ' **') 
        util.mkdir(plot_d+cut_dir)
        align_compare.align_residuals_vs_order(al_dirs, firstorder=orders[0], only_stars_in_fit=only_stars_in_fit, plot_dir=plot_d+cut_dir)
        print('\n')

    plt.close('all')
    

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

def epoch_quiver(target, epoch, epoch_num, align='align/align_t', poly='polyfit_d/fit', points='points_d/',
               useAccFits=False, magCut=18, root='./'):
    from astropy.io import fits
    from matplotlib.colors import LogNorm

    os.chdir(root)
    align_dir = os.getcwd()
    print(align_dir)
    print('*******************')

    data_root = '/u/jlu/data/microlens/'
    combo = data_root + epoch + '/combo/mag' + epoch + '_' + target + '_kp'
    img = fits.getdata(combo + '.fits')
    
    s = starset.StarSet(root + align)
    s.loadStarsUsed()
    s.loadPolyfit(root + poly, accel=0, arcsec=0)

    ee = epoch_num
    
    try: 
        pointsFile = root + points + target + '.points'
        if os.path.exists(pointsFile + '.orig'):
            pointsTab = Table.read(pointsFile + '.orig', format='ascii')
        else:
            pointsTab = Table.read(pointsFile, format='ascii')
            
        times = pointsTab[pointsTab.colnames[0]]
    except:
        print( 'Star ' + target + ' not in list' )
    
    plt.clf()
    plt.close(1)
    plt.figure(1, figsize=(10, 10))
    
    # Data
    x = s.getArrayFromEpoch(ee, 'xpix')
    y = s.getArrayFromEpoch(ee, 'ypix')
    m = s.getArrayFromEpoch(ee, 'mag')
    isUsed = s.getArrayFromEpoch(ee, 'isUsed')
    rad = np.hypot(x - 512, y - 512)

    good = np.where(isUsed == True)
    stars = s.stars

    Nstars = len(x)
    x_fit = np.zeros(Nstars, dtype=float)
    y_fit = np.zeros(Nstars, dtype=float)
    residsX = np.zeros(Nstars, dtype=float)
    residsY = np.zeros(Nstars, dtype=float)
    idx2 = []
    for i in range(Nstars):
        fitx = stars[i].fitXv
        fity = stars[i].fitYv 
        StarName = stars[i].name

        dt = times[ee] - fitx.t0
        fitLineX = fitx.p + (fitx.v * dt)
            
        fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

        fitLineY = fity.p + (fity.v * dt)
        fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

        x_fit[i] = fitLineX
        y_fit[i] = fitLineY
        residsX[i] = x[i] - fitLineX
        residsY[i] = y[i] - fitLineY
        
    idx = np.where((np.abs(residsX) < 10.0) & (np.abs(residsY) < 10.0))[0]
    print ("Trimmed {0:d} stars with too-large residuals (>10 pix)".format(len(idx)))
    plt.imshow(img, cmap='afmhot', norm=LogNorm(vmin=0.01,vmax=100000))
    plt.ylim(0, 1100)
    plt.xlim(0, 1100)
    plt.yticks(fontsize=10)
    plt.xticks([200,400,600,800,1000], fontsize=10)
    q = plt.quiver(x_fit, y_fit, residsX, residsY, scale_units='width', scale=0.5, color='gray')
    #q = plt.quiver(x_fit[idx], y_fit[idx], residsX[idx], residsY[idx], scale_units='width', scale=0.5, color='black')
    q = plt.quiver(x_fit[good], y_fit[good], residsX[good], residsY[good], scale_units='width', scale=0.5, color='red')
    plt.quiver([850, 0], [100, 0], [0.05, 0.05], [0, 0], color='red', scale=0.5, scale_units='width')
    plt.text(850, 120, '0.5 mas', color='red', fontsize=10)

    alignment = align_dir[-26:]
    plt.title(alignment + '_' + epoch)
    
    plt.show()
    plt.savefig(root + 'plots/' + epoch + '_quiver.png')
