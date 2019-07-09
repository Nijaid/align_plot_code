import numpy as np
from astropy.table import Table
import copy

def fit_velocities(table, bootstrap=0, verbose=False):
    """
    Fit velocities for all stars in the table.
    """

    table = copy.deepcopy(table)

    N_stars, N_epochs = table['x'].shape

    if verbose:
        start_time = time.time()
        msg = 'Starting startable.fit_velocities for {0:d} stars with n={1:d} bootstrap'
        print(msg.format(N_stars, bootstrap))

    # Clean/remove up old arrays.
    if 'x0' in table.colnames: table.remove_column('x0')
    if 'vx' in table.colnames: table.remove_column('vx')
    if 'y0' in table.colnames: table.remove_column('y0')
    if 'vy' in table.colnames: table.remove_column('vy')
    if 'x0e' in table.colnames: table.remove_column('x0e')
    if 'vxe' in table.colnames: table.remove_column('vxe')
    if 'y0e' in table.colnames: table.remove_column('y0e')
    if 'vye' in table.colnames: table.remove_column('vye')
    if 't0' in table.colnames: table.remove_column('t0')
    if 'n_vfit' in table.colnames: table.remove_column('n_vfit')

    # Define output arrays for the best-fit parameters.
    table.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'x0'))
    table.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vx'))
    table.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'y0'))
    table.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vy'))

    table.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'x0e'))
    table.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vxe'))
    table.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'y0e'))
    table.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vye'))

    table.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 't0'))
    table.add_column(Column(data = np.zeros(N_stars, dtype=int), name = 'n_vfit'))

    # Catch the case when there is only a single epoch. Just return 0 velocity
    # and the same input position for the x0/y0.
    if table['x'].shape[1] == 1:
        table['x0'] = table['x'][:,0]
        table['y0'] = table['y'][:,0]

        if 't' in table.colnames:
            table['t0'] = table['t'][:, 0]
        else:
            table['t0'] = table.meta['list_times'][0]

        if 'xe' in table.colnames:
            table['x0e'] = table['xe'][:,0]
            table['y0e'] = table['ye'][:,0]

        table['n_vfit'] = 1

        return

    # STARS LOOP through the stars and work on them 1 at a time.
    # This is slow; but robust.
    for ss in range(N_stars):
        table = fit_velocity_for_star(ss, bootstrap=bootstrap)

    if verbose:
        stop_time = time.time()
        print('startable.fit_velocities runtime = {0:.0f} s for {1:d} stars'.format(stop_time - start_time, N_stars))

    return table

def fit_velocity_for_star(table, ss, bootstrap=False):
    def poly_model(time, *params):
        pos = np.polynomial.polynomial.polyval(time, params)
        return pos

    x = table['x'][ss, :].data
    y = table['y'][ss, :].data

    if 'xe' in table.colnames:
        xe = table['xe'][ss, :].data
        ye = table['ye'][ss, :].data
    else:
        xe = np.ones(N_epochs, dtype=float)
        ye = np.ones(N_epochs, dtype=float)

    if 't' in table.colnames:
        t = table['t'][ss, :].data
    else:
        t = table.meta['list_times']

    # Figure out where we have detections (as indicated by error columns
    good = np.where((xe != 0) & (ye != 0) &
                    np.isfinite(xe) & np.isfinite(ye) &
                    np.isfinite(x) & np.isfinite(y))[0]

    N_good = len(good)

    # Catch the case where there is NO good data.
    if N_good == 0:
        return

    # Everything below has N_good >= 1
    x = x[good]
    y = y[good]
    t = t[good]
    xe = xe[good]
    ye = ye[good]

    # np.polynomial ordering
    p0x = np.array([x.mean(), 0.0])
    p0y = np.array([y.mean(), 0.0])

    # Calculate the t0 for all the stars.
    t_weight = 1.0 / np.hypot(xe, ye)
    t0 = np.average(t, weights=t_weight)
    dt = t - t0

    table['t0'][ss] = t0
    table['n_vfit'][ss] = N_good

    # Catch the case where all the times are identical
    if (dt == dt[0]).all():
        wgt_x = (1.0/xe)**2
        wgt_y = (1.0/ye)**2

        table['x0'][ss] = np.average(x, weights=wgt_x)
        table['y0'][ss] = np.average(y, weights=wgt_y)
        table['x0e'][ss] = np.sqrt(np.average((x - table['x0'][ss])**2, weights=wgt_x))
        table['y0e'][ss] = np.sqrt(np.average((y - table['y0'][ss])**2, weights=wgt_x))

        table['vx'][ss] = 0.0
        table['vy'][ss] = 0.0
        table['vxe'][ss] = 0.0
        table['vye'][ss] = 0.0

        return


    # Catch the case where we have enough measurements to actually
    # fit a velocity!
    if N_good > 2:
        vx_opt, vx_cov = curve_fit(poly_model, dt, x, p0=p0x, sigma=xe)
        vy_opt, vy_cov = curve_fit(poly_model, dt, y, p0=p0y, sigma=ye)

        table['x0'][ss] = vx_opt[0]
        table['vx'][ss] = vx_opt[1]
        table['y0'][ss] = vy_opt[0]
        table['vy'][ss] = vy_opt[1]

        # Run the bootstrap
        if bootstrap > 0:
            edx = np.arange(N_good, dtype=int)

            fit_x0_b = np.zeros(bootstrap, dtype=float)
            fit_vx_b = np.zeros(bootstrap, dtype=float)
            fit_y0_b = np.zeros(bootstrap, dtype=float)
            fit_vy_b = np.zeros(bootstrap, dtype=float)

            for bb in range(bootstrap):
                bdx = np.random.choice(edx, N_good)

                vx_opt_b, vx_cov_b = curve_fit(poly_model, dt[bdx], x[bdx], p0=vx_opt, sigma=xe[bdx])
                vy_opt_b, vy_cov_b = curve_fit(poly_model, dt[bdx], y[bdx], p0=vy_opt, sigma=ye[bdx])

                fit_x0_b[bb] = vx_opt_b[0]
                fit_vx_b[bb] = vx_opt_b[1]
                fit_y0_b[bb] = vy_opt_b[0]
                fit_vy_b[bb] = vy_opt_b[1]

            # Save the errors from the bootstrap
            table['x0e'][ss] = fit_x0_b.std()
            table['vxe'][ss] = fit_vx_b.std()
            table['y0e'][ss] = fit_y0_b.std()
            table['vye'][ss] = fit_vy_b.std()
        else:
            vx_err = np.sqrt(vx_cov.diagonal())
            vy_err = np.sqrt(vy_cov.diagonal())

            table['x0e'][ss] = vx_err[0]
            table['vxe'][ss] = vx_err[1]
            table['y0e'][ss] = vy_err[0]
            table['vye'][ss] = vy_err[1]

    elif N_good == 2:
        # Note nough epochs to fit a velocity.
        table['x0'][ss] = np.average(x, weights=1.0/xe**2)
        table['y0'][ss] = np.average(y, weights=1.0/ye)

        dx = np.diff(x)[0]
        dy = np.diff(y)[0]
        dt_diff = np.diff(dt)[0]

        table['x0e'][ss] = np.abs(dx) / 2**0.5
        table['y0e'][ss] = np.abs(dy) / 2**0.5
        table['vx'][ss] = dx / dt_diff
        table['vy'][ss] = dy / dt_diff
        table['vxe'][ss] = 0.0
        table['vye'][ss] = 0.0

    else:
        # N_good == 1 case
        table['n_vfit'][ss] = 1
        table['x0'][ss] = x[0]
        table['y0'][ss] = y[0]

        if 'xe' in table.colnames:
            table['x0e'] = xe[0]
            table['y0e'] = ye[0]

    return table
