import matplotlib
import matplotlib.style
import matplotlib.cm
matplotlib.use('agg')
matplotlib.style.use('paperstyle.mpl')
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import scipy.stats

import gr_xs

def format_axis(ax, xlims=None, ylims=None):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    intfloor = lambda x: int(np.floor(np.log10(x)))

    # Override the yaxis tick settings
    if ylims is not None:
        ymin, ymax = ylims
        ymin, ymax = intfloor(ymin), intfloor(ymax)
        major = 10.0 ** np.arange(ymin, ymax)
        minor = np.arange(2, 10) / 10.0
        locmaj = matplotlib.ticker.FixedLocator(10.0 ** np.arange(ymin, ymax))
        locmin = matplotlib.ticker.FixedLocator(
            np.tile(minor, len(major)) * np.repeat(major, len(minor))
        )
        locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1,), numticks=12)
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=minor, numticks=12)
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)
    if xlims is not None:
        xmin, xmax = xlims
        xmin, xmax = intfloor(xmin), intfloor(xmax)
        major = 10.0 ** np.arange(xmin, xmax)
        minor = np.arange(2, 10) / 10.0
        locmaj = matplotlib.ticker.FixedLocator(10.0 ** np.arange(xmin, xmax))
        locmin = matplotlib.ticker.FixedLocator(
            np.tile(minor, len(major)) * np.repeat(major, len(minor))
        )
        locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1,), numticks=12)
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=minor, numticks=12)
        ax.xaxis.set_major_locator(locmaj)
        ax.xaxis.set_minor_locator(locmin)

pb = 1.0e-36*(1.0e-2*5.06773093741e6)**2
pb = 1e-36
GeV = 1.0e9

cc = np.loadtxt('nusigma_sigma_CC.dat')
n_cc_energy = cc.shape[0]
cc_energy = cc[:,0]
cc_lines = cc[:,1:].T
nc = np.loadtxt('nusigma_sigma_NC.dat')
n_nc_energy = cc.shape[0]
nc_energy = nc[:,0]
nc_lines = nc[:,1:].T

linestyles = ['-', '--', '-.', ':']
cm = plt.get_cmap('plasma')
ncolors = 3
colors = [cm(x) for x in np.linspace(0.2, 0.9, ncolors)]
fig, ax = plt.subplots(figsize=(7,5))
flavors = [r'e', r'\mu', r'\tau']
bars = [r'', r'\bar']
labels = [b+r'\nu_'+f for f in flavors for b in bars]

nc_lines = np.array([np.sum(nc_lines[::2], axis=0)/3.0, np.sum(nc_lines[1::2], axis=0)/3.0])
artist_NC = ax.plot(nc_energy, nc_lines[0], color=colors[-2], linestyle='solid')
ax.plot(nc_energy, nc_lines[1], color=colors[-2], linestyle='dashed')

artist_solid = matplotlib.lines.Line2D([0], [0], color='black', linestyle='solid')
artist_dashed = matplotlib.lines.Line2D([0], [0], color='black', linestyle='dashed')

artists = []
artist_labels = []
artists.append(artist_solid)
artist_labels.append(r'$\nu$')
artists.append(artist_dashed)
artist_labels.append(r'$\bar\nu$')


cc_lines = np.array([np.sum(cc_lines[::2], axis=0)/3.0, np.sum(cc_lines[1::2], axis=0)/3.0])

linestyles = ['-', '--']
for i in range(2):
    artist = ax.plot(cc_energy, cc_lines[i], color=colors[int(i/2)], linestyle=linestyles[i%len(linestyles)])
    if i%2 == 0:
        artists.append(matplotlib.lines.Line2D([0],[0], color=colors[int(i/2)], linestyle=linestyles[i%len(linestyles)]))
        artist_labels.append(r'CC')

artists.append(matplotlib.lines.Line2D([0],[0], color=colors[-2], linestyle='solid'))
artist_labels.append(r'NC')

gr_rest = np.array([gr_xs.sigma_erest(e) for e in cc_energy])*1e9/1e36
gr_dopp_H = np.array([gr_xs.sigma_edopp(e, element='H') for e in cc_energy])*1e9/1e36
gr_dopp_O = np.array([gr_xs.sigma_edopp(e, element='O') for e in cc_energy])*1e9/1e36
gr_dopp_H2O = (8*gr_dopp_O + 2*gr_dopp_H)/10
#artist_GR = ax.plot(cc_energy, gr_rest, color='black', linestyle='-')
artist_GR = ax.plot(cc_energy, gr_dopp_H2O, color=colors[-1], linestyle='-.')

artists.append(matplotlib.lines.Line2D([0],[0], color=colors[-1], linestyle='-.'))
artist_labels.append(r'GR')

format_axis(ax, ylims=(1e-1/1e36, 1e6/1e36), xlims=(1e2, 1e9))
ax.set_xlabel('Neutrino Energy [GeV]')
ax.set_ylabel(r'Cross Section $[\textrm{cm}^2]$')
ax.legend(artists, artist_labels, frameon=True)
fig.tight_layout()
plt.savefig('xs.pdf')



