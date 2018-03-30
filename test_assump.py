import numpy as np
from microlens.jlu import model
from microlens.jlu.tests import test_model
from gcwork import starset
from astropy import units, constants
from astropy.table import Table
import pylab as py
import os
import pdb

def assume(t0, beta, tau, i0, ang, mL, dL, dS, source_m):
    """
    A quick model for microlensing signals based on the assumptions that the lens
    is 3 solar masses and that the distance to the lens is half of the distance to
    the source (2 kpc).

    t0 - float: Time of peak photometric signal in MJD.
    beta - float: Minimum angular distance between source and lens in mas.
    tau - float: Einstein crossing time in days.
    outdir - str: Path to directory where the plots will be saved.
    """
    xS0 = np.array([0.0, 0.0])
    muS = np.array([0.0, 0.0])
    muL = np.array([0.0, 0.0])
    angr = np.deg2rad(ang)

    # estimate source proper motion from Einstein radius and crossing time
    G = 6.67408e-11 # the big G [kg^-1 m^3 s^-2]
    L = dL*units.pc
    S = dS*units.pc
    M = mL*units.M_sun
    L = L.to('m').value
    S = S.to('m').value
    M = M.to('kg').value
    inv_dist = (1.0 / L) - (1.0 / S)
    thetaE = units.rad * np.sqrt((4.0 * G * M / 299792458.0**2) * inv_dist)
    Er = thetaE.to('mas').value
    print('Estimated Einstein radius: %f [mas]' %Er)
    
    mu_i = Er/tau*365.25
    if source_m:
        muS[0] = mu_i*np.cos(angr)
        muS[1] = mu_i*np.sin(angr)
        print('Estimated source proper motion: %f [mas/yr]' %np.linalg.norm(muS))
        print('mu_S = (%f, %f)' %(muS[0], muS[1]))
    else:
        muL[0] = mu_i*np.cos(angr)
        muL[1] = mu_i*np.sin(angr)
        print('Estimated lens proper motion: %f [mas/yr]' %np.linalg.norm(muL))
        print('mu_L = (%f, %f)' %(muL[0], muL[1]))

    test= '/mL_%.2f_dL_%.2f_dS_%.2f_ang_%.1f/' %(mL,dL/1000,dS/1000,ang)
    modeled = model.PSPL(mL, t0, xS0, beta, muL, muS, dL, dS, i0)

    return(test, modeled)

def AlignModel(t0, beta, tau, i0, target, align, source_m=True, ang=0.0, mL=1.0, dL=4000.0, dS=8000.0, save_fig=False, outdir='./'):
    '''
    Creates a PSPL model based on input assumptions against aligned data.

    Args:
        t0 - float: Time of peak photometric signal in days (t=0 on New Year's of year 0, UTC).
        beta - float: Minimum angular distance between source and lens in mas.
        tau - float: Einstein crossing time in days.
        i0 - float: Base photometric magnitude.
        target - str: Name of the target.
        align - str: Path to the alignment directory.
        source_m - bool: Determine if either the source is the moving object (True) or the lens (False).
        ang - float: Angle (degrees) at which the moving object travels w.r.t. the ecliptic.
        mL - float: Mass of the lens in solar masses.
        dL - float: Distance to the lens in parsecs.
        dS - float: Distance to the source in parsecs.
        save_fig - bool: To save or not to save the figure.
        outdir - str: Path to save figure. Folder is created in outdir.
    '''
    test, modeled = assume(t0, beta, tau, i0, ang, mL, dL, dS, source_m)
    
    if save_fig==True:
        os.chdir(outdir)
        test_dir = (outdir + test)
        print('Saving in ' + test_dir)
        if os.path.exists(test_dir) == False:
            os.mkdir(test_dir)

    # get the aligned data
    s = starset.StarSet(align + 'align/align_t')
    s.loadPolyfit(align + 'polyfit_d/fit', accel=0, arcsec=0)

    names = s.getArray('name')
    ss = names.index(target)
    star = s.stars[ss]

    pointsTab = Table.read(align + 'points_d/' + target + '.points', format='ascii')
    at = pointsTab[pointsTab.colnames[0]]
    ax = pointsTab[pointsTab.colnames[1]]
    ay = pointsTab[pointsTab.colnames[2]]
    axerr = pointsTab[pointsTab.colnames[3]]
    ayerr = pointsTab[pointsTab.colnames[4]]
    fitx = star.fitXv
    fity = star.fitYv

    mt = np.arange(t0-1500, t0+1500, 1)
    mdt = mt - modeled.t0
    adt = at - (modeled.t0 / 365.25)
    fdt = at - fitx.t0

    thE = modeled.thetaE_amp
    mshift = modeled.get_centroid_shift(mt)
    fitLineX = fitx.p + (fitx.v * fdt)
    fitSigX = np.sqrt( fitx.perr**2 + (fdt * fitx.verr)**2 )
    fitLineY = fity.p + (fity.v * fdt)
    fitSigY = np.sqrt( fity.perr**2 + (fdt * fity.verr)**2 )

    adt *= 365.25

    axf = (ax - fitLineX)
    ayf = (ay - fitLineY)
    
    ## plot everything scaled in Einstein units
    # x data
    alignment = align[-27:-1]
    fig = py.figure(figsize=(20,10))
    xpl = py.subplot(211) 
    py.plot(mdt / modeled.tE, mshift[:,0] / thE, 'k-')
    py.errorbar(adt / modeled.tE, axf*9.95 / thE, yerr=axerr*9.95 / thE, fmt='ro')
    py.legend(['model (tE = %.2f days)' %(modeled.tE), alignment], loc=4)
    py.plot(adt / modeled.tE, fitSigX * 9.95 / thE, 'b--')
    py.plot(adt / modeled.tE, -fitSigX * 9.95 / thE, 'b--')
    xpl.set_ylabel(r'dX / $\theta_E$')
    py.title(test[1:-1] + r' - $\theta_E =$ %.2f [mas]' %thE)

    # y data
    ypl = py.subplot(212, sharex=xpl)
    py.subplots_adjust(hspace=0)
    py.plot(mdt / modeled.tE, mshift[:,1] / thE, 'k-')
    py.plot(adt / modeled.tE, fitSigY * 9.95 / thE, 'b--')
    py.plot(adt / modeled.tE, -fitSigY * 9.95 / thE, 'b--')
    py.errorbar(adt / modeled.tE, ayf*9.95 / thE, yerr=ayerr*9.95 / thE, fmt='ro')
    ypl.set_ylabel(r'dY / $\theta_E$')
    ypl.set_xlabel('(t - t0) / tE')

    if save_fig==True:
        py.savefig(test_dir + 'shift_v_t.png')
    py.show()

    ## zoomed-in plot with residuals
    # approximate residual to integer model times
    madt = np.round(adt)
    xr = []
    yr = []
    for i in range(len(madt)):
        idx = np.where(mdt == madt[i])[0]
        xr.append(axf[i]*9.95 - mshift[idx,0])
        yr.append(ayf[i]*9.95 - mshift[idx,1])
    xr = np.array(xr)
    yr = np.array(yr)

    fig2, (xplz, xres, yplz, yres) = py.subplots(4,1, figsize=(15,15), gridspec_kw = {'height_ratios':[3,1,3,1]}, sharex=True)
    fig2.subplots_adjust(hspace=0)
    
    xplz.plot(mdt / modeled.tE, mshift[:,0] / thE, 'k-')
    xplz.errorbar(adt / modeled.tE, axf*9.95 / thE, yerr=axerr*9.95 / thE, fmt='ro')
    xplz.legend(['model (tE = %.2f days)' %(modeled.tE), alignment], loc=4)
    xplz.plot(adt / modeled.tE, fitSigX * 9.95 / thE, 'b--')
    xplz.plot(adt / modeled.tE, -fitSigX * 9.95 / thE, 'b--')
    xplz.set_ylabel(r'dX / $\theta_E$')
    xplz.set_title(test[1:-1] + r' - $\theta_E =$ %.2f [mas]' %thE)
    xres.errorbar(adt / modeled.tE, xr / thE, yerr=axerr*9.95 / thE, fmt='ro')
    xres.axhline(color='k')
    xres.set_ylabel('dX res')

    yplz.plot(mdt / modeled.tE, mshift[:,1] / thE, 'k-')
    yplz.plot(adt / modeled.tE, fitSigY * 9.95 / thE, 'b--')
    yplz.plot(adt / modeled.tE, -fitSigY * 9.95 / thE, 'b--')
    yplz.errorbar(adt / modeled.tE, ayf*9.95 / thE, yerr=ayerr*9.95 / thE, fmt='ro')
    yplz.set_xlim([adt[0] / modeled.tE - 2, adt[len(adt)-1] / modeled.tE + 2])
    yplz.set_ylabel(r'dY / $\theta_E$')
    yres.errorbar(adt / modeled.tE, yr / thE, yerr=ayerr*9.95 / thE, fmt='ro')
    yres.axhline(color='k')
    yres.set_ylabel('dY res')
    yres.set_xlabel('(t - t0) / tE')
    
    if save_fig==True:
        py.savefig(test_dir + 'shift_v_t_zoom.png')
    py.show()
      
def PhotoModel(t0, beta, tau, i0, stars, align, source_m=True, ang=0.0, mL=1.0, dL=4000.0, dS=8000.0, save_fig=False, outdir='./'):
    '''
    Creates a PSPL model based on input assumptions against aligned data.

    Args:
        t0 - float: Time of peak photometric signal in days (t=0 on New Year's of year 0, UTC).
        beta - float: Minimum angular distance between source and lens in mas.
        tau - float: Einstein crossing time in days.
        i0 - float: Base photometric magnitude.
        stars - str array: Array of stars.
        align - str: Path to the alignment directory.
        source_m - bool: Determine if either the source is the moving object (True) or the lens (False).
        ang - float: Angle (degrees) at which the moving object travels w.r.t. the ecliptic.
        mL - float: Mass of the lens in solar masses.
        dL - float: Distance to the lens in parsecs.
        dS - float: Distance to the source in parsecs.
        save_fig - bool: To save or not to save the figure.
        outdir - str: Path to save figure. Folder is created in outdir.
    '''
    test, modeled = assume(t0, beta, tau, i0, ang, mL, dL, dS, source_m)
    
    if save_fig==True:
        os.chdir(outdir)
        test_dir = (outdir + test)
        print('Saving in ' + test_dir)
        if os.path.exists(test_dir) == False:
            os.mkdir(test_dir)

    # get the model and aligned data
    s = starset.StarSet(align + 'align/align_t')
    s.loadPolyfit(align + 'polyfit_d/fit', accel=0, arcsec=0)

    # x = s.getArrayFromAllEpochs('xpix')
    # y = s.getArrayFromAllEpochs('ypix')
    # xe_p = s.getArrayFromAllEpochs('xpixerr_p')
    # ye_p = s.getArrayFromAllEpochs('ypixerr_p')

    N_epochs = len(s.years)
    mags = np.zeros([N_epochs, 3])
    mean_mag = np.zeros([3,1])

    for i in range(len(stars)):
        photTab = Table.read(align + 'points_d/' + stars[i] + '.phot', format='ascii')
        mags[:,i] = photTab[photTab.colnames[6]]
        if i == 0:
            m0 = photTab[photTab.colnames[6]]
        mean_mag[i] = np.mean(mags[:,i])
        mags[:,i] -= mean_mag[i]

    at = photTab[photTab.colnames[0]]
    mt = np.arange(t0-1500, t0+1500, 1)
    mdt = mt - modeled.t0
    adt = at - (modeled.t0 / 365.25)

    adt *= 365.25

    #get OGLE data
    ogle = Table.read('/g/lu/microlens/cross_epoch/OB150211/OGLE-2015-BLG-0211.dat', format='ascii')
    ot = (ogle['col1'] - 1721057.5)/365.25
    om = ogle['col2']
    omr = ogle['col3']

    # get the modeled photometry
    photo = modeled.get_photometry(at*365.25)
    
    fig1 = py.figure(figsize=(8,8))
    py.errorbar(ot, om, yerr=omr, fmt='o')
    py.plot(at, photo, 'ko', at, m0+7.83, 'ro')
    py.ylabel('mag')
    py.legend(['model', 'uncalibrated data + 7.83', 'OGLE O-IV optimized'])
    py.tight_layout()
    if save_fig:
        py.savefig(test_dir + 'model_phot.png')
    py.show()
    
    fig2 = py.figure(figsize=(8,8))
    py.plot(at, mags[:,0], 'ko', at, mags[:,1], 'o', at, mags[:,2], 'o')
    py.legend([stars[0], stars[1], stars[2]])
    py.ylabel(r'$\Delta$mag')
    py.tight_layout()
    if save_fig:
        py.savefig(test_dir + 'del_phot.png')
    py.show()

    return photo
    
