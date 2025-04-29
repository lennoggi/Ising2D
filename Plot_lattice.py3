import numpy as np
import h5py
from matplotlib import pyplot as plt
import argparse


# Set up the parser
description = "Plot the lattice at all available iterations"
parser      = argparse.ArgumentParser(description = description)

parser.add_argument("--infile", required = True, type = str,
                    help = "HDF5 file containining the lattice data")
parser.add_argument("--plotdir", required = True, type = str,
                    help = "path to the directory where the snapshots will be saved")

args    = parser.parse_args()
infile  = args.infile
plotdir = args.plotdir


# Plot
f = h5py.File(infile, "r")

for dset_name in list(f.keys()):
    dset  = f[dset_name]

    plt.figure(figsize = (10, 10), dpi = 200)
    plt.axis("off")
    plt.imshow(dset, origin = "lower", interpolation = "none", aspect = "equal",
               cmap = "plasma", vmin = -1., vmax = 1.)
    plt.tight_layout(pad = 0.)
    figname = f"{plotdir}/{dset_name}.png"
    plt.savefig(figname)
    plt.close()

    print(f"File {figname} generated successfully")
