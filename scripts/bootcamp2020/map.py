import matplotlib
import matplotlib.style
matplotlib.use('agg')
matplotlib.style.use('paperstyle.mpl')
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import scipy.stats

loc_0 = 2.
scale_0 = 0.5
mass_0 = 10.

loc_1 = 3.
scale_1 = scale_0/10.
mass_1 = 1.

f = lambda x: (sp.stats.norm.pdf(x, loc=loc_0, scale=scale_0)*mass_0+sp.stats.norm.pdf(x, loc=loc_1, scale=scale_1)*mass_1)/(mass_0+mass_1)
f_cdf = lambda x: (sp.stats.norm.cdf(x, loc=loc_0, scale=scale_0)*mass_0+sp.stats.norm.cdf(x, loc=loc_1, scale=scale_0)*mass_1)/(mass_0+mass_1)
x = np.linspace(0,5,10000+1)
y = f(x)
x0 = x[np.argmax(y)]

fig, ax = plt.subplots(figsize=(7,5))
ax.fill_between(x, y)
ax.set_xlim((0.0, 5.0))
ax.set_ylim((0.0, None))
plt.savefig('dist.png')

fig, ax = plt.subplots(figsize=(7,5))
ax.fill_between(x, y)
ax.axvline(x0, color='black', linestyle='dashed')
ax.set_xlim((0.0, 5.0))
ax.set_ylim((0.0, None))
plt.savefig('map.png')


def find_bounds(level, x, y):
    b0 = y == level
    e_i = np.arange(len(b0))[b0]
    b1 = y <= level
    b2 = y >= level
    b3 = np.logical_and(~np.equal(b1[:-1], b1[1:]), ~np.equal(b2[:-1], b2[1:]))
    i = np.arange(len(b3))[b3]
    i = np.sort(np.unique(np.concatenate([i, e_i])))
    x_places = []
    for j in i:
        y0, y1 = y[j], y[j+1]
        x0, x1 = x[j], x[j+1]
        xx = x0 + (x1-x0)/(y1-y0)*(level-y0)
        x_places.append(xx)
    x_places = [0] + x_places + [5]
    x_places = np.sort(x_places)
    is_over = []
    x_segments = []
    y_segments = []
    total_over = 0.0
    for j in range(len(x_places)-1):
        mask = np.logical_and(x >= x_places[j], x <= x_places[j+1])
        is_over.append(np.all(y[mask] >= level))
        x_segments.append(np.concatenate([[x_places[j]], x[mask], [x_places[j+1]]]))
        y_segments.append(np.concatenate([[level], y[mask], [level]]))
        if is_over[j]:
            total_over += f_cdf(x_places[j+1]) - f_cdf(x_places[j])
    return total_over, is_over, x_segments, y_segments

fig, ax = plt.subplots(figsize=(7,5))
#ax.fill_between(x, y)
tots = []
levels = np.linspace(0, 1, 10000+1)
for level in levels:
    total_over, _, _, _ = find_bounds(level, x, y)
    tots.append(total_over)
tots = np.array(tots)
mass = 0.683
i = np.argmin(np.abs(tots - mass))
level = levels[i]
total_over, is_over, x_segments, y_segments = find_bounds(level, x, y)
ax.axhline(level, color='black', linestyle='dashed')
for j in range(len(is_over)):
    if is_over[j]:
        ax.fill_between(x_segments[j], y_segments[j], color='#ff7f0e', edgecolor='none', linewidth=0)
    else:
        ax.fill_between(x_segments[j], y_segments[j], color='#1f77b4', edgecolor='none', linewidth=0)
ax.set_xlim((0.0, 5.0))
ax.set_ylim((0.0, None))
plt.savefig('hpd.png')
