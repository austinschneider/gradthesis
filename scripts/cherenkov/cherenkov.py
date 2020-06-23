import matplotlib
import matplotlib.style
import matplotlib.cm
import matplotlib.collections
import matplotlib.patches
matplotlib.use('agg')
matplotlib.style.use('paperstyle.mpl')
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import scipy.stats


c = 1
def get_x(t, beta=0.999):
    return beta*c*t

def get_r(t, n=1.76):
    return c*t/n

def circles(x, y, s, c='b', ax=None, vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of circles.
    Similar to plt.scatter, but the size of circles are in data scale.
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circles.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    a = np.arange(11)
    circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    plt.colorbar()
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    zipped = np.broadcast(x, y, s)
    patches = [matplotlib.patches.Circle((x_, y_), s_)
               for x_, y_, s_ in zipped]
    collection = matplotlib.collections.PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection

beta=0.999
fig, ax = plt.subplots()
ax.axis('equal')
current_t = 10
t = np.linspace(0, 10, 10)
circles(get_x(t, beta=beta), 0, get_r(current_t-t), ax=ax, fc='none', ec='blue', lw=1)

# Wave fronts
lines = np.array([(-2, 0), (0, 0)])
line_collection = matplotlib.collections.LineCollection([lines], color='red', linestyle='dashed', lw=1, joinstyle='miter')
ax.add_collection(line_collection)

# Muon arrow
lines = np.array([(0, 0), (get_x(current_t, beta=beta), 0)])
ax.arrow(0,0,get_x(current_t, beta=beta),0, color='red', linestyle='solid', lw=1, head_width=0.2, overhang=0.2, length_includes_head=True)

max_t = np.amax(current_t-t)
r = get_r(max_t)
x = get_x(max_t, beta=beta)
h = np.sqrt(x**2-r**2)
b = r*h/x
l = r**2/x
y0 = b/(1.-l/x)

# Tangent line
lines = [(0, y0), (get_x(current_t, beta=beta), 0)]
line_collection = matplotlib.collections.LineCollection([lines], color='black', linestyle='solid', lw=1, joinstyle='miter')
ax.add_collection(line_collection)

# Box line
lines = [(0, 0), (l, b)]
line_collection = matplotlib.collections.LineCollection([lines], color='black', linestyle='solid', lw=1, joinstyle='miter')
ax.add_collection(line_collection)
s = 0.5
m0 = b/l
m1 = -l/b
th0 = np.arctan2(b,l)
th1 = np.pi/2.-th0
x0 = abs(s*np.cos(th0))
y0 = abs(s*np.sin(th0))
x1 = abs(s*np.cos(th1))
y1 = abs(s*np.sin(th1))
p0 = (l, b)
p1 = (l-x0, b-y0)
p2 = (p1[0]+x1, p1[1]-y1)
p3 = (p2[0]+x0, p2[1]+y0)
lines = [[p0, p1, p2, p3, p0]]
#lines = [[(l*0.95, b*0.95), (l*0.95+0.05*b, b*0.95-0.05*l), (l+0.05*b, b-0.05*l), (l*0.95+0.05*b, b*0.95-0.05*l)]]
line_collection = matplotlib.collections.LineCollection(lines, color='black', linestyle='solid', lw=1, joinstyle='miter')
ax.add_collection(line_collection)

# arc
patch = matplotlib.patches.Arc((0,0), get_r(max_t)*0.25, get_r(max_t)*0.25, angle=0, theta1=0, theta2=np.arccos(get_r(max_t)/get_x(max_t, beta=beta))*180./np.pi, edgecolor='black', linewidth=1, fill=False)
ax.add_patch(patch)
ax.annotate(r'$\theta_c$', xy=(get_r(max_t)*0.2*np.cos(th0/2.),get_r(max_t)*0.2*np.sin(th0/2.)), ha='left', va='center')


# Distance travelled
lines = [[(0, 0), (0,-get_r(current_t)*1.1)],[(get_x(current_t, beta=beta),0), (get_x(current_t, beta=beta), -get_r(current_t)*1.1)]]
line_collection = matplotlib.collections.LineCollection(lines, color='black', linestyle='dashed', lw=1, joinstyle='miter')
ax.add_collection(line_collection)
lines = [(0,-get_r(current_t)*1.1), (get_x(current_t, beta=beta), -get_r(current_t)*1.1)]
ax.arrow(lines[0][0], lines[0][1], lines[1][0]-lines[0][0], lines[1][1]-lines[0][1], color='black', linestyle='solid', lw=1, head_width=0.2, overhang=0.2, length_includes_head=True)
ax.arrow(lines[1][0], lines[1][1], lines[0][0]-lines[1][0], lines[0][1]-lines[1][1], color='black', linestyle='solid', lw=1, head_width=0.2, overhang=0.2, length_includes_head=True)
ax.annotate(r'$\beta c t$', xy=(get_x(current_t, beta=beta)/2., -get_r(current_t)*1.2), ha='center', va='center')

# Distance propagated
lines = [[(get_r(current_t),0), (get_r(current_t), -get_r(current_t)*0.9)]]
line_collection = matplotlib.collections.LineCollection(lines, color='black', linestyle='dashed', lw=1, joinstyle='miter')
ax.add_collection(line_collection)
lines = [(0,-get_r(current_t)*0.9), (get_r(current_t), -get_r(current_t)*0.9)]
ax.arrow(lines[0][0], lines[0][1], lines[1][0]-lines[0][0], lines[1][1]-lines[0][1], color='black', linestyle='solid', lw=1, head_width=0.2, overhang=0.2, length_includes_head=True)
ax.arrow(lines[1][0], lines[1][1], lines[0][0]-lines[1][0], lines[0][1]-lines[1][1], color='black', linestyle='solid', lw=1, head_width=0.2, overhang=0.2, length_includes_head=True)
ax.annotate(r'$ct/n$', xy=(get_r(current_t)/2., -get_r(current_t)), ha='center', va='center')
ax.set_ylim((-7.5,7.5))
ax.axis('off')
#fig.tight_layout()
#plt.savefig('cherenkov.pdf', bbox_inches='tight')
plt.savefig('cherenkov.pdf')


################

beta=0.4
fig, ax = plt.subplots()
ax.axis('equal')
current_t = 10
t = np.linspace(0, 10, 10)
circles(get_x(t, beta=beta), 0, get_r(current_t-t), ax=ax, fc='none', ec='blue', lw=1)

# Wave fronts
lines = np.array([(-2, 0), (0, 0)])
line_collection = matplotlib.collections.LineCollection([lines], color='red', linestyle='dashed', lw=1, joinstyle='miter')
ax.add_collection(line_collection)

# Muon arrow
lines = np.array([(0, 0), (get_x(current_t, beta=beta), 0)])
ax.arrow(0,0,get_x(current_t, beta=beta),0, color='red', linestyle='solid', lw=1, head_width=0.2, overhang=0.2, length_includes_head=True)

max_t = np.amax(current_t-t)
r = get_r(max_t)
x = get_x(max_t, beta=beta)
h = np.sqrt(x**2-r**2)
b = r*h/x
l = r**2/x
y0 = b/(1.-l/x)

# Distance travelled
lines = [[(0, 0), (0,-get_r(current_t)*1.1)],[(get_x(current_t, beta=beta),0), (get_x(current_t, beta=beta), -get_r(current_t)*1.1)]]
line_collection = matplotlib.collections.LineCollection(lines, color='black', linestyle='dashed', lw=1, joinstyle='miter')
ax.add_collection(line_collection)
lines = [(0,-get_r(current_t)*1.1), (get_x(current_t, beta=beta), -get_r(current_t)*1.1)]
ax.arrow(lines[0][0], lines[0][1], lines[1][0]-lines[0][0], lines[1][1]-lines[0][1], color='black', linestyle='solid', lw=1, head_width=0.2, overhang=0.2, length_includes_head=True)
ax.arrow(lines[1][0], lines[1][1], lines[0][0]-lines[1][0], lines[0][1]-lines[1][1], color='black', linestyle='solid', lw=1, head_width=0.2, overhang=0.2, length_includes_head=True)
ax.annotate(r'$\beta c t$', xy=(get_x(current_t, beta=beta)/2., -get_r(current_t)*1.2), ha='center', va='center')

# Distance propagated
lines = [[(get_r(current_t),0), (get_r(current_t), -get_r(current_t)*0.9)]]
line_collection = matplotlib.collections.LineCollection(lines, color='black', linestyle='dashed', lw=1, joinstyle='miter')
ax.add_collection(line_collection)
lines = [(0,-get_r(current_t)*0.9), (get_r(current_t), -get_r(current_t)*0.9)]
ax.arrow(lines[0][0], lines[0][1], lines[1][0]-lines[0][0], lines[1][1]-lines[0][1], color='black', linestyle='solid', lw=1, head_width=0.2, overhang=0.2, length_includes_head=True)
ax.arrow(lines[1][0], lines[1][1], lines[0][0]-lines[1][0], lines[0][1]-lines[1][1], color='black', linestyle='solid', lw=1, head_width=0.2, overhang=0.2, length_includes_head=True)
ax.annotate(r'$ct/n$', xy=(get_r(current_t)/2., -get_r(current_t)), ha='center', va='center')
ax.set_ylim((-7.5,7.5))
ax.axis('off')
#fig.tight_layout()
#plt.savefig('no_cherenkov.pdf', bbox_inches='tight')
plt.savefig('no_cherenkov.pdf')
