import matplotlib
import matplotlib.style
import matplotlib.cm
matplotlib.use('agg')
matplotlib.style.use('paperstyle.mpl')
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import scipy.stats

f = lambda x: scipy.stats.chi2.pdf(x, 1)
f_cdf = lambda x: scipy.stats.chi2.cdf(x, 1)
x = np.linspace(0,5,10000+1)
y = f(x)
ts_i = np.argmin(abs(f_cdf(x)-0.95))

fig, ax = plt.subplots(figsize=(7,5))
ax.axvline(x[ts_i], color='black', linestyle='dashed', label=r'$\textrm{TS}_\textrm{obs}$')
ax.fill_between(x[ts_i:], y[ts_i:])
ax.plot(x, y, label=r'$\mathcal{P}(\textrm{TS}|\theta)$')
ax.set_yscale('log')
ax.set_xlim((0, 5))
ax.set_xlabel('TS')
ax.set_ylabel(r'$\mathcal{P}(\textrm{TS}|\theta)$')
ax.legend(frameon=True)
fig.tight_layout()
plt.savefig('ts_dist.pdf')



x_obs_0 = x[ts_i]
x_obs_1 = x_obs_0/2.
x_obs_2 = x_obs_1/2.
f0 = lambda x: scipy.stats.chi2.pdf(x, 1)
f0_cdf = lambda x: scipy.stats.chi2.cdf(x, 1)
x = np.linspace(0,5,10000+1)
y0 = f0(x)

p0 = 1.0-f0_cdf(x_obs_0)
p1 = 1.0-f0_cdf(x_obs_1)
p2 = 1.0-f0_cdf(x_obs_2)

cm = matplotlib.cm.get_cmap('plasma')
colors = [cm(0.1 + (i/3.)*0.8) for i in range(3)]

fig, ax = plt.subplots(figsize=(7,5))
ax.axvline(x_obs_0, color=colors[0], linestyle='dashed', label=r'$\textrm{TS}_\textrm{obs}(\theta_0), p_\textrm{value}=%.02f$' % p0)
ax.axvline(x_obs_1, color=colors[1], linestyle='dashed', label=r'$\textrm{TS}_\textrm{obs}(\theta_1), p_\textrm{value}=%.02f$' % p1)
ax.axvline(x_obs_2, color=colors[2], linestyle='dashed', label=r'$\textrm{TS}_\textrm{obs}(\theta_2), p_\textrm{value}=%.02f$' % p2)
ax.plot(x, y0, label=r'$\mathcal{P}(\textrm{TS}|\theta)$', color='black')
ax.set_yscale('log')
ax.set_xlim((0, 5))
ax.set_xlabel('TS')
ax.set_ylabel(r'$\mathcal{P}(\textrm{TS}|\theta)$')
ax.legend(frameon=True, ncol=2)
fig.tight_layout()
plt.savefig('ts_func_dist.pdf')


x_obs_0 = x[ts_i]
x_obs_1 = x_obs_0/2.
x_obs_2 = x_obs_1/2.
f0 = lambda x: scipy.stats.chi2.pdf(x, 1)
f0_cdf = lambda x: scipy.stats.chi2.cdf(x, 1)
f1 = lambda x: (scipy.stats.chi2.pdf(x, 1) + scipy.stats.chi2.pdf(x, 2))/2.
f1_cdf = lambda x: (scipy.stats.chi2.cdf(x, 1) + scipy.stats.chi2.cdf(x, 2))/2.
f2 = lambda x: scipy.stats.chi2.pdf(x, 2)
f2_cdf = lambda x: scipy.stats.chi2.cdf(x, 2)
x = np.linspace(0,5,10000+1)
y0 = f0(x)
y1 = f1(x)
y2 = f2(x)

p0 = 1.0-f0_cdf(x_obs_0)
p1 = 1.0-f1_cdf(x_obs_1)
p2 = 1.0-f2_cdf(x_obs_2)

cm = matplotlib.cm.get_cmap('plasma')
colors = [cm(0.1 + (i/3.)*0.8) for i in range(3)]

fig, ax = plt.subplots(figsize=(7,5))
ax.axvline(x_obs_0, color=colors[0], linestyle='dashed', label=r'$\textrm{TS}_\textrm{obs}(\theta_0), p_\textrm{value}=%.02f$' % p0)
ax.axvline(x_obs_1, color=colors[1], linestyle='dashed', label=r'$\textrm{TS}_\textrm{obs}(\theta_1), p_\textrm{value}=%.02f$' % p1)
ax.axvline(x_obs_2, color=colors[2], linestyle='dashed', label=r'$\textrm{TS}_\textrm{obs}(\theta_2), p_\textrm{value}=%.02f$' % p2)
ax.plot(x, y0, label=r'$\mathcal{P}(\textrm{TS}|\theta_0)$', color=colors[0])
ax.plot(x, y1, label=r'$\mathcal{P}(\textrm{TS}|\theta_1)$', color=colors[1])
ax.plot(x, y2, label=r'$\mathcal{P}(\textrm{TS}|\theta_2)$', color=colors[2])
ax.set_yscale('log')
ax.set_xlim((0, 5))
ax.set_xlabel('TS')
ax.set_ylabel(r'$\mathcal{P}(\textrm{TS}|\theta)$')
ax.legend(frameon=True, ncol=2)
fig.tight_layout()
plt.savefig('ts_func_dists.pdf')
