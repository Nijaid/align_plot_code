import matplotlib.pyplot as plt
import numpy as np
import os

t = np.logspace(-2, 2, 10000)
pi_rel = 4.848e-9*(1./4.0 - 1./8.0)*3600.*1000.*180./np.pi
thE = np.sqrt(8.1459*5.0*pi_rel)
u = np.sqrt(t*t + 0.01)
shift = 10.0 * thE * u / (u*u + 2.0)
amp = (u*u + 2.0) / (u * np.sqrt(u*u + 4.0))
print("theta_E = %.3f"%thE)

fig = plt.figure(1)
ax = plt.axes()
ax.set_xscale('log')
ax.grid(True, which='both', linestyle=':', color='grey')

plt.plot(t, amp, 'k--', label='amplification')
plt.plot(t, shift, 'k', label='centroid shift x10 (mas)')
plt.xlabel(r'$\left( t - t_0 \right)\; / \; t_E$')
plt.legend()
plt.xlim(t[0], t[-1])
if os.path.exists('../paper_plots/'):
    plt.savefig('../paper_plots/shift_mag_5.pdf', bbox_inches='tight')
else:
    out_dir = raw_input("Where would you like to save? ")
    plt.savefig(out_dir + '/shift_mag_5.pdf', bbox_inches='tight')

plt.close(fig)
