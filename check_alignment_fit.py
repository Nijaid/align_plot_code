import shutil, os, sys
import pylab as py
import numpy as np
import scipy
import scipy.stats
from gcwork import starset
from gcwork import starTables
from astropy.table import Table
from jlu.util import fileUtil
import sys
import pdb
import scipy.stats

def align_plot(root_dir='./', align_root='align'):
    s = starset.StarSet(root_dir + align_root)
    s.loadStarsUsed()

    x = s.getArrayFromAllEpochs('xpix')
    y = s.getArrayFromAllEpochs('ypix')
    xe_p = s.getArrayFromAllEpochs('xpixerr_p')
    ye_p = s.getArrayFromAllEpochs('ypixerr_p')
    xe_a = s.getArrayFromAllEpochs('xpixerr_a')
    ye_a = s.getArrayFromAllEpochs('ypixerr_a')
    isUsed = s.getArrayFromAllEpochs('isUsed')

    x0 = s.getArray('fitpXv.p')
    y0 = s.getArray('fitpYv.p')
    vx = s.getArray('fitpXv.v')
    vy = s.getArray('fitpYv.v')
    t0x = s.getArray('fitpXv.t0')
    t0y = s.getArray('fitpYv.t0')

    m = s.getArray('mag')
    cnt = s.getArray('velCnt')

    N_epochs = x.shape[0]
    N_stars = x.shape[1]

    # Setup arrays
    xresid_rms_all = np.zeros(N_epochs, dtype=float)
    yresid_rms_all = np.zeros(N_epochs, dtype=float)

    xresid_rms_used = xresid_rms_all
    yresid_rms_used = xresid_rms_all

    xresid_err_a_all = xresid_rms_all
    yresid_err_a_all = xresid_rms_all

    xresid_err_a_used = xresid_rms_all
    yresid_err_a_used = xresid_rms_all

    xresid_err_p_all = xresid_rms_all
    yresid_err_p_all = xresid_rms_all

    xresid_err_p_used = xresid_rms_all
    yresid_err_p_used = xresid_rms_all

    chi2x_all = xresid_rms_all
    chi2y_all = xresid_rms_all

    chi2x_used = xresid_rms_all
    chi2y_used = xresid_rms_all

    N_stars_all = xresid_rms_all
    N_stars_used = xresid_rms_all

    year = xresid_rms_all

    idx = np.where(cnt > 2)[0]
    N_stars_3ep = len(idx)

    for ee in range(N_epochs):
        idx = np.where((cnt > 2) & (x[ee, :] > -1000) & (xe_p[ee, :] > 0))[0]
        used = np.where(isUsed[ee, idx] == True)

        # Everything below should be arrays sub-indexed by "idx"
        dt_x = s.years[ee] - t0x[idx]
        dt_y = s.years[ee] - t0y[idx]

        x_fit = x0[idx] + (vx[idx] * dt_x)
        y_fit = y0[idx] + (vy[idx] * dt_y)

        xresid = x[ee, idx] - x_fit
        yresid = y[ee, idx] - y_fit

        N_stars_all[ee] = len(xresid)
        N_stars_used[ee] = len(xresid[used])

        # Note this chi^2 only includes positional errors.
        chi2x_terms = xresid**2 / xe_p[ee, idx]**2
        chi2y_terms = yresid**2 / ye_p[ee, idx]**2

        xresid_rms_all[ee] = np.sqrt(np.mean(xresid**2))
        yresid_rms_all[ee] = np.sqrt(np.mean(yresid**2))

        xresid_rms_used[ee] = np.sqrt(np.mean(xresid[used]**2))
        yresid_rms_used[ee] = np.sqrt(np.mean(yresid[used]**2))

        xresid_err_p_all[ee] = xe_p[ee, idx].mean() / N_stars_all[ee]**0.5
        yresid_err_p_all[ee] = ye_p[ee, idx].mean() / N_stars_all[ee]**0.5

        xresid_err_p_used[ee] = xe_p[ee, idx][used].mean() / N_stars_used[ee]**0.5
        yresid_err_p_used[ee] = ye_p[ee, idx][used].mean() / N_stars_used[ee]**0.5

        xresid_err_a_all[ee] = xe_a[ee, idx].mean() / N_stars_all[ee]**0.5
        yresid_err_a_all[ee] = ye_a[ee, idx].mean() / N_stars_all[ee]**0.5

        xresid_err_a_used[ee] = xe_a[ee, idx][used].mean() / N_stars_used[ee]**0.5
        yresid_err_a_used[ee] = ye_a[ee, idx][used].mean() / N_stars_used[ee]**0.5

        chi2x_all[ee] = chi2x_terms.sum()
        chi2y_all[ee] = chi2y_terms.sum()

        chi2x_used[ee] = chi2x_terms[used].sum()
        chi2y_used[ee] = chi2y_terms[used].sum()

        year[ee] = s.years[ee]

    data = {'xres_rms_all': xresid_rms_all, 'yres_rms_all': yresid_rms_all,
            'xres_rms_used': xresid_rms_used, 'yres_rms_used': yresid_rms_used,
            'xres_err_p_all': xresid_err_p_all, 'yres_err_p_all': yresid_err_p_all,
            'xres_err_p_used': xresid_err_p_used, 'yres_err_p_used': yresid_err_p_used,
            'xres_err_a_all': xresid_err_a_all, 'yres_err_a_all': yresid_err_a_all,
            'xres_err_a_used': xresid_err_a_used, 'yres_err_a_used': yresid_err_a_used,
            'chi2x_all': chi2x_all, 'chi2y_all': chi2y_all,
            'chi2x_used': chi2x_used, 'chi2y_used': chi2y_used,
            'N_stars_all': N_stars_all, 'N_stars_used': N_stars_used,
            'year': year, 'N_stars_3ep': N_stars_3ep}

    return data
