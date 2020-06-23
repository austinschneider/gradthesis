import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def cartesian_product_simple_transpose(arrays):
    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T

energy = (lambda x: (x[:-1]+x[1:])/2.)(np.logspace(0,11,89)) # in GeV
distance = 1./np.logspace(np.log10(1./1000), np.log10(1./(2e5)), 100) # in m
energy_bins = np.logspace(0,11,89) # in GeV
distance_diff = (np.log10(1./1000) - np.log10(1./(2e5))) / (100-1)
distance_bins = 1./np.logspace(np.log10(1./1000)-distance_diff/2., np.log10(1./(2e5))+distance_diff/2., 100+1) # in m

f = open('output.txt', 'r')
entries = np.zeros((len(energy), len(distance), 0)).tolist()
initial_energies = []
distances = []
final_energies = []
for line in f:
    items = line.split(' ')
    energy_i = int(items[0])
    distance_i = int(items[1])
    final_energy = float(items[2])
    if distance_i == len(distance):
        continue
    if energy_i == len(energy):
        continue
    entries[energy_i][distance_i].append(final_energy)
    initial_energies.append(energy[energy_i])
    distances.append(distance[distance_i])
    final_energies.append(final_energy)
initial_energies = np.array(initial_energies)
distances = np.array(distances)
final_energies = np.array(final_energies)

entries = np.array(entries)

def prep_2d_hist(x, y, x_edges, y_edges, quantity, f=None):
    x_mapping = np.digitize(x, bins=x_edges) - 1
    y_mapping = np.digitize(y, bins=y_edges) - 1
    n_x_edges = len(x_edges)
    n_y_edges = len(y_edges)
    n_x_bins = n_x_edges-1
    n_y_bins = n_y_edges-1
    print(n_x_bins, n_y_bins)
    #bin_masks = np.empty((n_y_bins, n_x_bins, len(x)))
    binned_quantity = np.empty((n_y_bins, n_x_bins, 0)).tolist()
    binned_result = np.empty((n_y_bins, n_x_bins)).tolist()
    for j in range(n_y_bins):
	for k in range(n_x_bins):
            mask = np.logical_and(x_mapping == k, y_mapping == j)
	    #bin_masks[j,k,:] = mask
            binned_quantity[j][k] = quantity[mask]
            if f is not None:
                binned_result[j, k] = f(binned_quantity[j][k])
    X = np.array([x_edges]*(n_y_edges))
    Y = np.array([y_edges]*(n_x_edges)).T
    if f is not None:
        return X, Y, binned_quantity, binned_result
    else:
        return X, Y, binned_quantity

X, Y, binned_q = prep_2d_hist(initial_energies, distances, energy_bins, distance_bins, final_energies)

median = np.array([[np.median(x) for x in l] for l in binned_q])
print(median)

cm = plt.get_cmap('plasma')
fig, ax = plt.subplots(figsize=(7,5))
mesh = ax.pcolormesh(X,Y,median, cmap=cm, norm=matplotlib.colors.LogNorm(vmin=np.amin(median[median>0]), vmax=np.amax(median)))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Initial Muon Energy [GeV]')
ax.set_ylabel(r'Muon Overburden [m]')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Median Final Energy')
cb.ax.minorticks_off()
plt.tight_layout()
fig.savefig('./preach_median.pdf')
fig.clf()
plt.close(fig)
