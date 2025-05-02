import numpy as np
import h5py
import re
from matplotlib import pyplot as plt
import argparse


# Compile the regex once (see routine below) to improve performance.
# Also, [0-9]+ does not enforce the to-be-matched string ends after the integer(s).
it_pattern = re.compile(r"Iteration_(\d+)")

# Helper routine to sort the iterations in the file in numerical order
# (0, 1, 2, 3, ...) instead of lexicographical order (0, 10, 11, 12, ...)
def extract_iteration(instr):
    m = it_pattern.match(instr)
    if m: return int(m.group(1))  # m.group(0) is the full string "instr", m.group(1) is the iteration number
    else: raise RuntimeError(f"String '{instr}' does not match the expected format 'Iteration_xxxx'")


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
with h5py.File(infile, "r") as f:
    for dset_name in sorted(list(f.keys()), key = extract_iteration):
        dset  = f[dset_name]

        plt.figure(figsize = (10., 10.), dpi = 200)
        plt.axis("off")
        plt.imshow(dset, interpolation = "none", aspect = "equal",
                   cmap = "plasma", vmin = -1, vmax = 1)
        plt.tight_layout(pad = 0.)
        figname = f"{plotdir}/{dset_name}.png"
        plt.savefig(figname)
        plt.close()

        print(f"File {figname} generated successfully")
