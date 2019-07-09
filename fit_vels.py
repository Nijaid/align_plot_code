import numpy as np
from astropy.table import Table

class StarTable(Table):
    def __init__(self, table_):
        tab = Table.read(table_)
        Table.__init__(self, tab)

        return
        
    def fit_velocities(self, bootstrap=0, verbose=False):
        """
        Fit velocities for all stars in the self.
        """

        N_stars, N_epochs = self['x'].shape

        if verbose:
            start_time = time.time()
            msg = 'Starting startable.fit_velocities for {0:d} stars with n={1:d} bootstrap'
            print(msg.format(N_stars, bootstrap))

        # Clean/remove up old arrays.
        if 'x0' in self.colnames: self.remove_column('x0')
        if 'vx' in self.colnames: self.remove_column('vx')
        if 'y0' in self.colnames: self.remove_column('y0')
        if 'vy' in self.colnames: self.remove_column('vy')
        if 'x0e' in self.colnames: self.remove_column('x0e')
        if 'vxe' in self.colnames: self.remove_column('vxe')
        if 'y0e' in self.colnames: self.remove_column('y0e')
        if 'vye' in self.colnames: self.remove_column('vye')
        if 't0' in self.colnames: self.remove_column('t0')
        if 'n_vfit' in self.colnames: self.remove_column('n_vfit')

        # Define output arrays for the best-fit parameters.
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'x0'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vx'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'y0'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vy'))

        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'x0e'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vxe'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'y0e'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vye'))

        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 't0'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=int), name = 'n_vfit'))

        # Catch the case when there is only a single epoch. Just return 0 velocity
        # and the same input position for the x0/y0.
        if self['x'].shape[1] == 1:
            self['x0'] = self['x'][:,0]
            self['y0'] = self['y'][:,0]

            if 't' in self.colnames:
                self['t0'] = self['t'][:, 0]
            else:
                self['t0'] = self.meta['list_times'][0]

            if 'xe' in self.colnames:
                self['x0e'] = self['xe'][:,0]
                self['y0e'] = self['ye'][:,0]

            self['n_vfit'] = 1

            return

        # STARS LOOP through the stars and work on them 1 at a time.
        # This is slow; but robust.
        for ss in range(N_stars):
            self.fit_velocity_for_star(ss, bootstrap=bootstrap)

        if verbose:
            stop_time = time.time()
            print('startable.fit_velocities runtime = {0:.0f} s for {1:d} stars'.format(stop_time - start_time, N_stars))

        return

    def fit_velocity_for_star(self, ss, bootstrap=False):
        def poly_model(time, *params):
            pos = np.polynomial.polynomial.polyval(time, params)
            return pos

        x = self['x'][ss, :].data
        y = self['y'][ss, :].data

        if 'xe' in self.colnames:
            xe = self['xe'][ss, :].data
            ye = self['ye'][ss, :].data
        else:
            xe = np.ones(N_epochs, dtype=float)
            ye = np.ones(N_epochs, dtype=float)

        if 't' in self.colnames:
            t = self['t'][ss, :].data
        else:
            t = self.meta['list_times']

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

        self['t0'][ss] = t0
        self['n_vfit'][ss] = N_good

        # Catch the case where all the times are identical
        if (dt == dt[0]).all():
            wgt_x = (1.0/xe)**2
            wgt_y = (1.0/ye)**2

            self['x0'][ss] = np.average(x, weights=wgt_x)
            self['y0'][ss] = np.average(y, weights=wgt_y)
            self['x0e'][ss] = np.sqrt(np.average((x - self['x0'][ss])**2, weights=wgt_x))
            self['y0e'][ss] = np.sqrt(np.average((y - self['y0'][ss])**2, weights=wgt_x))

            self['vx'][ss] = 0.0
            self['vy'][ss] = 0.0
            self['vxe'][ss] = 0.0
            self['vye'][ss] = 0.0

            return


        # Catch the case where we have enough measurements to actually
        # fit a velocity!
        if N_good > 2:
            vx_opt, vx_cov = curve_fit(poly_model, dt, x, p0=p0x, sigma=xe)
            vy_opt, vy_cov = curve_fit(poly_model, dt, y, p0=p0y, sigma=ye)

            self['x0'][ss] = vx_opt[0]
            self['vx'][ss] = vx_opt[1]
            self['y0'][ss] = vy_opt[0]
            self['vy'][ss] = vy_opt[1]

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
                self['x0e'][ss] = fit_x0_b.std()
                self['vxe'][ss] = fit_vx_b.std()
                self['y0e'][ss] = fit_y0_b.std()
                self['vye'][ss] = fit_vy_b.std()
            else:
                vx_err = np.sqrt(vx_cov.diagonal())
                vy_err = np.sqrt(vy_cov.diagonal())

                self['x0e'][ss] = vx_err[0]
                self['vxe'][ss] = vx_err[1]
                self['y0e'][ss] = vy_err[0]
                self['vye'][ss] = vy_err[1]

        elif N_good == 2:
            # Note nough epochs to fit a velocity.
            self['x0'][ss] = np.average(x, weights=1.0/xe**2)
            self['y0'][ss] = np.average(y, weights=1.0/ye)

            dx = np.diff(x)[0]
            dy = np.diff(y)[0]
            dt_diff = np.diff(dt)[0]

            self['x0e'][ss] = np.abs(dx) / 2**0.5
            self['y0e'][ss] = np.abs(dy) / 2**0.5
            self['vx'][ss] = dx / dt_diff
            self['vy'][ss] = dy / dt_diff
            self['vxe'][ss] = 0.0
            self['vye'][ss] = 0.0

        else:
            # N_good == 1 case
            self['n_vfit'][ss] = 1
            self['x0'][ss] = x[0]
            self['y0'][ss] = y[0]

            if 'xe' in self.colnames:
                self['x0e'] = xe[0]
                self['y0e'] = ye[0]

        return
