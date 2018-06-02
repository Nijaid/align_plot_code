import numpy as np
import matplotlib.pyplot as py
from astropy.table import Table
from nirc2.reduce import util
from microlens.jlu import model, model_fitter
import pdb


def getdata(target, pdir):
    phot = Table.read('/g/lu/microlens/cross_epoch/OB150211/OGLE-2015-BLG-0211.dat', format='ascii')
    points = Table.read(pdir + target + '.points', format='ascii')

    points['col2'] = (points['col2'] - points['col2'][0]) * 0.00995
    points['col3'] = (points['col3'] - points['col3'][0]) * 0.00995
    points['col4'] *= 0.00995
    points['col5'] *= 0.00995

    data = {'mag_err': phot['col3'],
            'mag': phot['col2'],
            'xpos': points['col2'],
            'ypos': points['col3'],
            'xpos_err': points['col4'],
            'ypos_err': points['col5'],
            't_ast': points['col1']*365.25 - 678943.0, # convert to MJD 
            't_phot': phot['col1'] - 2400000.5} # convert to MJD approximately (from OGLE's HJD)

    if target=='ob150211':
        data['raL'] = 17.4906056
        data['decL'] = -30.9817500
    else:
        ValueError(target+' does not have a listed RA and dec')

    return data

def modelfit(target, align_dir, solve = True, parallax = False, points_dir = 'points_d/', runcode = 'aa_'):
    data = getdata(target, align_dir+points_dir)

    if parallax == False:
        mdir = 'mnest_pspl/'
        fit = model_fitter.PSPL_Solver(data)
    elif parallax == True:
        mdir = 'mnest_pspl_par/'
        fit = model_fitter.PSPL_parallax_Solver(data)

    fit.outputfiles_basename = align_dir+mdir+'aa_'
    if solve:
        if target == 'ob150211':
            fit.mag_base_gen = model_fitter.make_gen(16.0, 18.0)
            fit.dL_gen = model_fitter.make_gen(500, 8000)
            
        util.mkdir(align_dir+mdir)
        
        fit.solve()
        fit.plot_posteriors()
    
    modeled = fit.get_best_fit_model()

    t_dat = np.linspace(np.min(data['t_phot']), np.max(data['t_phot']), num=len(data['t_phot'])*2, endpoint=True)
    mag_dat = modeled.get_photometry(t_dat)
    mag = modeled.get_photometry(data['t_phot'])
    pos = modeled.get_astrometry(data['t_ast'])

    lnL_phot = modeled.likely_photometry(data['t_phot'], data['mag'], data['mag_err'])
    lnL_ast = modeled.likely_astrometry(data['t_ast'], data['xpos'], data['ypos'], data['xpos_err'], data['ypos_err'])

    lnL = lnL_phot.mean() + lnL_ast.mean()
    print('lnL: ', lnL)

    # data['t_phot'] /= 365.25
    # data['t_ast'] /= 365.25

    util.mkdir(align_dir+mdir+'plots/')

    fig1, (pho, pho_res) = py.subplots(2,1, figsize=(10,10), gridspec_kw = {'height_ratios': [3,1]}, sharex=True)
    fig1.subplots_adjust(hspace=0)
    pho.errorbar(data['t_phot'], data['mag'], yerr=data['mag_err'], fmt='k.')
    pho.plot(t_dat, mag_dat, 'r-')
    pho.set_ylabel('mag')
    pho.invert_yaxis()
    pho.legend(['best-fit model', 'OGLE-IV'])
    pho.set_title(target + ' Photometry')
    pho_res.errorbar(data['t_phot'], data['mag'] - mag, yerr=data['mag_err'], fmt='k.')
    pho_res.plot(data['t_phot'], mag-mag, 'r-', lw=2)
    pho_res.invert_yaxis()
    pho_res.set_ylabel('data - model')
    pho_res.set_xlabel('days (MJD)')
    py.savefig(align_dir+mdir+'plots/photo.png')

    py.figure(2)
    py.clf()
    py.errorbar(data['t_ast'], data['xpos'], yerr=data['xpos_err'], fmt='k.')
    py.plot(data['t_ast'], pos[:,0], 'r-')
    py.xlabel('days (MJD)')
    py.ylabel('X Pos (")')
    py.legend(['model', 'aligned data'])
    py.title('X')
    py.savefig(align_dir+mdir+'plots/x_pos.png')

    py.figure(3)
    py.clf()
    py.errorbar(data['t_ast'], data['ypos'], yerr=data['ypos_err'], fmt='k.')
    py.plot(data['t_ast'], pos[:,1], 'r-')
    py.xlabel('days (MJD)')
    py.ylabel('Y Pos (")')
    py.legend(['model', 'aligned data'])
    py.title('Y')
    py.savefig(align_dir+mdir+'plots/y_pos.png')

    py.figure(4)
    py.clf()
    py.errorbar(data['xpos'], data['ypos'], xerr=data['xpos_err'], yerr=data['ypos_err'], fmt='k.')
    py.plot(pos[:,0], pos[:,1], 'r-')
    py.gca().invert_xaxis()
    py.xlabel('X Pos (")')
    py.ylabel('Y Pos (")')
    py.legend(['model', 'aligned data'])
    py.title(target + ' X and Y')
    py.savefig(align_dir+mdir+'plots/pos.png')

    #pdb.set_trace()
    best = Table(fit.get_best_fit())
    best.write(align_dir+'mnest_pspl/best_values.dat', format='ascii.fixed_width',
                   delimiter=' ', overwrite=True)
    
    return
