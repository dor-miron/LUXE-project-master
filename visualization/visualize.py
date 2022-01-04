import multiprocessing

from mayavi import mlab
import numpy as np
from scipy import stats

from data_loader.EcalDataIO import ecalmatio


def plot_density_3d(xyz, weights=None, multiprocess=False):
    """ coords of shape (samples, 3) """

    x, y, z = xyz[1, :], xyz[2, :], xyz[0, :]

    # Evaluate kde on a grid
    xmin, ymin, zmin = x.min(), y.min(), z.min()
    xmax, ymax, zmax = x.max(), y.max(), z.max()
    xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])

    kde = stats.gaussian_kde(xyz, 0.1)

    if multiprocess:
        def calc_kde(data):
            return kde(data.T)
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        results = pool.map(calc_kde, np.array_split(coords.T, 2))
        density = np.concatenate(results).reshape(xi.shape)
    else:
        density = kde(coords).reshape(xi.shape)

    # Plot scatter with mayavi
    figure = mlab.figure('DensityPlot')

    grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
    min = density.min()
    max = density.max()
    mlab.pipeline.volume(grid, vmin=min, vmax=min + .5*(max-min))

    mlab.axes()
    mlab.show()


def plot_ecal_file_3d(edep_path, energy_path, event_id):
    ecal = ecalmatio(edep_path)
    event = ecal[event_id]
    xyz_list = list(event.keys())
    xyz_array = np.array(xyz_list).T
    value_list = [event[v] for v in xyz_list]
    plot_density_3d(xyz_array, value_list)


def plot_density_3d_example(mu=0, sigma=0.1):
    x = 10 * np.random.normal(mu, sigma, 5000)
    y = 10 * np.random.normal(mu, sigma, 5000)
    z = 10 * np.random.normal(mu, sigma, 5000)

    xyz = np.vstack([x, y, z])
    plot_density_3d(xyz)


if __name__ == '__main__':
    # plot_density_3d_example(0, 0.2)

    edep_path = r"C:\Users\dor00\PycharmProjects\LUXE-project-master\data\raw\signal.al.elaser.IP05.edeplist.mat"
    energy_path = r"C:\Users\dor00\PycharmProjects\LUXE-project-master\data\raw\signal.al.elaser.IP05.energy.mat"
    plot_ecal_file_3d(edep_path, energy_path, '818')
