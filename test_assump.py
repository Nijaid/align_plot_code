import numpy as np
from jlu.microlens import test_model, model
from gcwork import starset
from astropy import units, constants, table
import pylab as py
import os
import pdb

def assume(t0, beta, tau, imag, outdir):
    """
    A quick model for microlensing signals based on the assumptions that the lens
    is 3 solar masses and that the distance to the lens is half of the distance to
    the source (2 kpc).

    t0 - float: Time of peak photometric signal in MJD.
    beta - float: Minimum angular distance between source and lens in mas.
    tau - float: Einstein crossing time in days.
    outdir - str: Path to directory where the plots will be produced.
    """
    mL = 1.0 # solar mass
    xS0 = np.array([0.0,0.0])
    dL = 4000.0 # pc
    dS = 2*dL
    muS = np.array([0.0, 0.0])
    muL = np.array([0.0, 0.0])

    # estimate source proper motion from Einstein radius and crossing time
    G = 6.67408e-11 # the big G
    L = dL*units.pc
    S = dS*units.pc
    M = mL*units.M_sun
    L = L.to('m').value
    S = S.to('m').value
    M = M.to('kg').value
    inv_dist = (1.0 / L) - (1.0 / S)
    thetaE = units.rad * np.sqrt((4.0 * G * M / 299792458.0**2) * inv_dist)
    Er = thetaE.to('mas').value
    muS[0] = Er/tau*365.25
    print('Estimated Einstein radius: %f [mas]' %Er)
    print('Estimated source proper motion: %f [mas/yr]' %muS[0])

    if os.path.exists(outdir) == False:
        os.mkdir(outdir)

    test_model.test_PSPL(mL, t0, xS0, beta, muL, muS, dL, dS, imag, outdir)
    py.close('all')

def ModelAlign(t0, beta, tau, imag, target, align, mL=1.0, dL=4000.0, dS=8000.0, root = '/u/nijaid/work/'):
    '''
    Compare a model based on input assumptions against aligned data.

    Args:
        t0 - float: Time of peak photometric signal in days (t=0 on New Year's of year 0, UTC).
        beta - float: Minimum angular distance between source and lens in mas.
        tau - float: Einstein crossing time in days.
        imag - float: Base photometric magnitude.
        target - str: Lowercase name of the target.
        align_dir - str: Path to the alignment directory.
        mL - float: Mass of the lens in solar masses.
        dL - float: Distance to the lens in parsecs.
        dS - float: Distance to the source in parsecs.
        root - string: Path to the work directory.
    '''
    xS0 = np.array([0.0, 0.0])
    muS = np.array([0.0, 0.0])
    muL = np.array([0.0, 0.0])

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
    muS[0] = Er/tau*365.25
    print('Estimated Einstein radius: %f [mas]' %Er)
    print('Estimated source proper motion: %f [mas/yr]' %muS[0])

    os.chdir(root + target.upper() + '/' + align)
    os.chdir('../tests/')
    test = ('/mL_%.1f_dL_%.1f_dS_%.1f/' %(mL,dL,dS))
    outdir = os.getcwd() + test
    print outdir
    if os.path.exists(outdir) == False:
        os.mkdir(outdir)

    # get the model and the aligned data
    modeled = model.PSPL(mL, t0, xS0, beta, muL, muS, dL, dS, imag)
    align_dir = root + target.upper() + '/' + align
    s = starset.StarSet(align_dir + 'align/align_t')
    s.loadPolyfit(align_dir + 'polyfit_d/fit', accel=0, arcsec=0)

    names = s.getArray('name')
    ss = names.index(target)
    star = s.stars[ss]

    pointsTab = table.Table.read(align_dir + 'points_d/' + target + '.points', format='ascii')
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

    thE = modeled.thetaE_amp
    mshift = modeled.get_centroid_shift(mt)
    fitLineX = fitx.p + (fitx.v * adt)
    fitSigX = np.sqrt( fitx.perr**2 + (adt * fitx.verr)**2 )
    fitLineY = fity.p + (fity.v * adt)
    fitSigY = np.sqrt( fity.perr**2 + (adt * fity.verr)**2 )

    adt *= 365.25

    # plot everything scaled in Einstein units
    alignment = align[35:-1]
    fig = py.figure(figsize=(20,10))

    xpl = py.subplot(211)
    py.plot(mdt / modeled.tE, mshift[:,0] / thE, 'k-')
    py.errorbar(adt / modeled.tE, (ax - fitLineX)*9.95 / thE, yerr=axerr*9.95 / thE, fmt='ro')
    py.legend(['model',alignment])
    py.plot(adt / modeled.tE, fitSigX * 9.95 / thE, 'b--')
    py.plot(adt / modeled.tE, -fitSigX * 9.95 / thE, 'b--')
    xpl.set_ylabel(r'dX / $\theta_E$')
    py.title(test[1:-1] + r' - $\theta_E =$ %.2f [mas]' %thE)

    ypl = py.subplot(212, sharex=xpl)
    py.subplots_adjust(hspace=0)
    py.plot(mdt / modeled.tE, mshift[:,1] / thE, 'k-')
    py.plot(adt / modeled.tE, fitSigY * 9.95 / thE, 'b--')
    py.plot(adt / modeled.tE, -fitSigY * 9.95 / thE, 'b--')
    py.errorbar(adt / modeled.tE, (ay - fitLineY)*9.95 / thE, yerr=ayerr*9.95 / thE, fmt='ro')
    ypl.set_ylabel(r'dY / $\theta_E$')
    ypl.set_xlabel('(t - t0) / tE')

    py.savefig(outdir + 'shift_v_t.png')

    #zoomed-in plot
    fig2 = py.figure(figsize=(10,10))

    xplz = py.subplot(211)
    py.plot(mdt / modeled.tE, mshift[:,0] / thE, 'k-')
    py.errorbar(adt / modeled.tE, (ax - fitLineX)*9.95 / thE, yerr=axerr*9.95 / thE, fmt='ro')
    py.legend(['model',alignment])
    py.plot(adt / modeled.tE, fitSigX * 9.95 / thE, 'b--')
    py.plot(adt / modeled.tE, -fitSigX * 9.95 / thE, 'b--')
    xplz.set_ylabel(r'dX / $\theta_E$')
    py.title(test[1:-1] + r' - $\theta_E =$ %.2f [mas]' %thE)

    yplz = py.subplot(212, sharex=xplz)
    py.subplots_adjust(hspace=0)
    py.plot(mdt / modeled.tE, mshift[:,1] / thE, 'k-')
    py.plot(adt / modeled.tE, fitSigY * 9.95 / thE, 'b--')
    py.plot(adt / modeled.tE, -fitSigY * 9.95 / thE, 'b--')
    py.errorbar(adt / modeled.tE, (ay - fitLineY)*9.95 / thE, yerr=ayerr*9.95 / thE, fmt='ro')
    yplz.set_xlim([-5, 5])
    yplz.set_ylabel(r'dY / $\theta_E$')
    yplz.set_xlabel('(t - t0) / tE')

    py.savefig(outdir + 'shift_v_t_zoom.png')
