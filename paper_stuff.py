import numpy as np
from math import ceil
import matplotlib.pyplot as py
from astropy.table import Table
from astropy.io import fits
from matplotlib.colors import LogNorm
from microlens.jlu import align_compare
from microlens.jlu import model, model_fitter
from gcwork import starset
import os
import pdb

def average(target):
    '''
    Find the averages and number of frames from the clean lists in each epoch.
    '''
    os.chdir('/u/jlu/data/microlens/')
    epochs = analyzed(target)[0]

    for ep in epochs:
        strehl = ep + '/clean/' + target + '_kp/strehl_source.txt'
        clean =  ep + '/combo/mag' + ep + '_' + target + '_kp.lis'

        # Create list of clean frames
        lis = open(clean, 'r')
        clean_lis = lis.read().split()
        frames =[]
        for i in range(len(clean_lis)):
            frames.append(clean_lis[i][-10:])
        lis.close()

        strehl_ = Table.read(strehl, format='ascii')
        strehl_tot = len(strehl_)

        for i in range(len(frames)):
            if strehl_['col1'][i] == frames[i]:
                pass
            else:
                strehl_.remove_row(i)

        dropped = strehl_tot - len(frames)
        print('*** Epoch: ' + ep,
                  '\nDropped/Total frames: %d/%d' %(dropped,strehl_tot),
                  '\nStrehl: ', strehl_['col2'].mean(), '\nRMS error (nm): ', strehl_['col3'].mean(),
                  '\nFWHM:', strehl_['col4'].mean(), '\n')

def field0211():    
    root = '/u/jlu/data/microlens/15jun07/combo/mag15jun07_'

    fig = py.figure(figsize=(5,5))
    img = fits.getdata(root + 'ob150211_kp.fits')
    coords = Table.read(root + 'ob150211_kp.coo', format='ascii')

    x = coords['col1']
    y = coords['col2']

    # Circle OB150211
    ax = py.gca().axes
    py.imshow(img, cmap='Greys', norm=LogNorm(vmin=10.0, vmax=10000))
    ax.add_artist(py.Circle((x,y), 70, color='#000080', fill=False))

    # Plot scale
    py.plot([150,351.1], [100,100], color='magenta', linewidth=2)
    py.text(238.5, 125, '2"', color='magenta', fontsize=10)

    # Plot compass
    py.text(1006, 1010, 'N', color='green', fontsize=11)
    ax.arrow(1022,917, 0,75, head_width=10, head_length=10, fc='green', ec='green')
    py.text(900, 904, 'E', color='green', fontsize=11)
    ax.arrow(1022,917, -75,0, head_width=10, head_length=10, fc='green', ec='green')

    py.xlim(30,1100)
    py.ylim(40,1060)
    ax.axis('off')

    py.savefig('/u/nijaid/microlens/paper_plots/0211field.png', dpi=200,
                   bbox_inches='tight')

def starfield(): # plot the targets in their 15jun07 image
    root = '/u/jlu/data/microlens/15jun07/combo/mag15jun07_'

    targets = ['OB140613','OB150029','OB150211']
    fig = py.figure(figsize=(15,5))
    for i in range(len(targets)):
        img = fits.getdata(root + targets[i] + '_kp.fits')
        coords = Table.read(root + targets[i] + '_kp.coo', format='ascii')
        x = coords['col1']
        y = coords['col2']

        py.subplot(1,3,i+1)
        py.imshow(img, cmap='Greys', norm=LogNorm(vmin=10.0,vmax=10000))
        # Draw line and label the target
        py.plot([x+100,x], [y-100,y], 'r-')
        py.text(x+100, y-150, targets[i], fontsize=12, color='red')

        # Plot scale
        py.plot([150,351.1], [100,100], color='magenta', linewidth=2)
        py.text(238.5, 125, '2"', color='magenta', fontsize=10)

        # Plot compass
        ax = py.gca().axes
        py.text(1006, 1010, 'N', color='green', fontsize=11)
        ax.arrow(1022,917, 0,75, head_width=10, head_length=10, fc='green', ec='green')
        py.text(900, 904, 'E', color='green', fontsize=11)
        ax.arrow(1022,917, -75,0, head_width=10, head_length=10, fc='green', ec='green')

        py.xlim(30,1100)
        py.ylim(40,1060)
        ax.axis('off')
        py.subplots_adjust(wspace=0.025)

    # py.show()
    py.savefig('/u/nijaid/microlens/paper_plots/starfield.png', dpi=300)

def mag_poserror(target, outdir='/u/nijaid/microlens/paper_plots/'):
    '''
    Plot magnitude v position uncertainty in epoch subplots.

    target - str: Lowercase name of the target.
    '''
    from microlens.jlu import analysis
    root = '/u/jlu/data/microlens/'
    epochs, xlim = analyzed(target)
    if target=='ob150211':
         an = analysis.OB150211(epochs[0], 'kp')
    if target=='ob140613':
        an = analysis.OB140613(epochs[0], 'kp')
    if target=='ob150029':
        an = analysis.OB150029(epochs[0], 'kp')
    magCutOff = an.plotPosMagCut
    radius = 4
    scale = 0.00995

    Nrows = int(ceil(float(len(epochs))/3))
    n = 1

    py.close('all')
    fig, axes = py.subplots(Nrows, 3, figsize=(10,8), sharex=True, sharey=True)
    for epoch in epochs:
        starlist = root + '%s/combo/starfinder/mag%s_%s_kp_rms.lis' %(epoch,epoch,target)
        print(starlist)

        lis = Table.read(starlist, format='ascii')
        name = lis[lis.colnames[0]]
        mag = lis[lis.colnames[1]]
        x = lis[lis.colnames[3]]
        y = lis[lis.colnames[4]]
        xerr = lis[lis.colnames[5]]
        yerr = lis[lis.colnames[6]]
        snr = lis[lis.colnames[7]]
        corr = lis[lis.colnames[8]]

        merr = 1.086 / snr

        # Convert into arsec offset from field center
        # We determine the field center by assuming that stars
        # are detected all the way out the edge.
        xhalf = x.max() / 2.0
        yhalf = y.max() / 2.0
        x = (x - xhalf) * scale
        y = (y - yhalf) * scale
        xerr *= scale * 1000.0
        yerr *= scale * 1000.0

        r = np.hypot(x, y)
        err = (xerr + yerr) / 2.0

        tar = np.where(name == target)[0]
        idx = (np.where((mag < magCutOff) & (r < radius)))[0]

        py.subplot(Nrows, 3, n)
        py.semilogy(mag[idx], err[idx], 'k.')
        py.semilogy(mag[tar], err[tar], 'r.') # plot the target in red
        py.axis(xlim)
        date = '20' + epoch[0:2] + ' ' + epoch[2].upper() + epoch[3:5] + ' ' + epoch[5:]
        py.text(xlim[0]+0.5, 10, date, fontsize=11)

        ax = py.gca().axes
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        if int(n%3.0) != 1:
            ax.yaxis.set_ticklabels([])
        if (n+3) <= len(epochs):
            ax.xaxis.set_ticklabels([])
        else:
            py.xlabel('Kp Magnitude', fontsize=12)

        n += 1

    for aa in range(3*Nrows): # delete empty subplots
        if aa >= len(epochs):
            fig.delaxes(axes.flatten()[aa])

    py.subplots_adjust(hspace=0.03, wspace=0.03)

    fig.add_subplot(111, frameon=False)
    py.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    py.ylabel('Positional Uncertainty (mas)', fontsize=12)

    out = outdir + target + '_magPosEpochs'
    py.savefig(out + '.eps', dpi=300, bbox_inches='tight')
    print('\nFigure saved in ' + outdir + '\n')

def align_res(target, date, prefix='a', Kcut=18, weight=4, transform=4, export=False, root='/u/nijaid/work/', main_out='/u/nijaid/microlens/paper_plots/'):
    work_dir = root + target.upper() + '/' + prefix + '_' + date + '/'

    align_dir = work_dir + prefix + '_' + target + '_' + date + '_a' + str(transform) + '_m' + str(Kcut) + '_w' + str(weight) + '_MC100/'
    s = starset.StarSet(align_dir + 'align/align_t')
    s.loadPolyfit(align_dir + 'polyfit_d/fit', accel=0, arcsec=0)

    names = s.getArray('name')
    i = names.index(target)
    star = s.stars[i]

    pointsTab = Table.read(align_dir + 'points_d/' + target + '.points', format='ascii')
    time = np.array(pointsTab[pointsTab.colnames[0]])
    x = pointsTab[pointsTab.colnames[1]]
    y = pointsTab[pointsTab.colnames[2]]
    xerr = pointsTab[pointsTab.colnames[3]] * 9.95
    yerr = pointsTab[pointsTab.colnames[4]] * 9.95

    fitx = star.fitXv
    fity = star.fitYv
    dt = time - fitx.t0
    fitLineX = fitx.p + (fitx.v*dt)
    fitLineY = fity.p + (fity.v*dt)

    resX = (x - fitLineX) * 9.95
    resY = (y - fitLineY) * 9.95

    fmt = '{0:.3f}: dx = {1:6.3f}  dy = {2:6.3f}  xe = {3:6.3f}  ye = {3:6.3f}'
    for yy in range(len(time)):
        print(fmt.format(time[yy], resX[yy], resY[yy], xerr[yy], yerr[yy]))

    if export:
        out_dir = main_out + 'align_res_' + target + '_' + date + '_a' + str(transform) + '_m' + str(Kcut) + '_w' + str(weight)
        _out = open(out_dir + '.txt', 'w')

        pfmt = fmt + '\n'
        for yy in range(len(time)):
            _out.write(pfmt.format(time[yy], resX[yy], resY[yy], xerr[yy], yerr[yy]))

        _out.close()


def analyzed(target):
    if target == 'ob150211':
        epochs = ['15may05', '15jun07', '15jun28', '15jul23', '16may03', '16jul14', '16aug02', '17jun05', '17jun08', '17jul19', '18may11']
        xlim = [8, 17.5, 1e-2, 30.0]
    elif target == 'ob150029':
        epochs = ['15jun07', '15jul23', '16may24', '16jul14', '17may21', '17jul14', '17jul19']
        xlim = [11.5, 22, 1e-2, 30.0]
    elif target == 'ob140613':
        epochs = ['15jun07', '15jun28', '16apr17', '16may24', '16aug02', '17jun05', '17jul14', '18may11']
        xlim = [11.75, 22.5, 1e-2, 30.0]
    else:
        raise ValueError(target + ' is not a listed target (try changing case)')
    return epochs, xlim


if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        if argv[1] == 'starfield':
            starfield()
        elif argv[1] == 'field0211':
            field0211()
        elif argv[1] == 'mag_poserror':
            mag_poserror(argv[2])
        elif argv[1] == 'average':
            average(argv[2])
        else:
            raise Exception('{} is not a valid command'.format(argv[1]))
