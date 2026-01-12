import numpy as np
import h5py
import argparse

# Parser setup
description = "Analyze the results of a 2D Ising lattice simulation"
parser      = argparse.ArgumentParser(description)

parser.add_argument("infile",  help = "Path to the HDF5 input file")
parser.add_argument("outfile", help = "Path to the HDF5 output file")
parser.add_argument("-v", type = int, required = False, default = -1,
                    help = "If positive, enable verbose output")

args      = parser.parse_args()
infile    = args.infile
outfile   = args.outfile
out_every = args.v


# Initialization
f = h5py.File(infile, "r")

beta     = f["Beta"][0]
nx1, nx2 = f["Lattice size"][()]
mag      = f["Magnetization"][()]
energy   = f["Energy"][()]
x1sums   = f["x1 spin sums"][()]
x2sums   = f["x2 spin sums"][()]

f.close()

vol = nx1*nx2
c1  = beta*vol
c2  = beta*c1

ncalc = len(mag)
if len(energy) != ncalc:
    raise RuntimeError("Inconsistent number of iterations in datasets 'Magnetization' and 'Energy'")
if x1sums.shape != (ncalc, nx1):
    raise RuntimeError("Inconsistent array length for spin sums along x1")
if x2sums.shape != (ncalc, nx2):
    raise RuntimeError("Inconsistent array length for spin sums along x2")



# Calculate the "time"-averaged magnetization and energy, magnetic
# susceptibility, and specific heat. Also, calculate the "time"-averaged spin
# sums over rows and columns to later compute the correlation function(s)
mag_av    = 0.
energy_av = 0.

sum_diffs_mag    = 0.
sum_diffs_energy = 0.

x1sums_av = np.zeros(nx1)
x2sums_av = np.zeros(nx2)

# NOTE: Welford online algorithm for mean and variance of magnetization and
#       energy
for n in range(ncalc):
    mag_n    = mag[n]
    energy_n = energy[n]

    delta1_mag    = mag_n    - mag_av
    delta1_energy = energy_n - energy_av

    mag_av    += delta1_mag/(1.*(n+1))
    energy_av += delta1_energy/(1.*(n+1))

    delta2_mag    = mag_n    - mag_av
    delta2_energy = energy_n - energy_av

    sum_diffs_mag    += delta1_mag*delta2_mag
    sum_diffs_energy += delta1_energy*delta2_energy

    for i in range(nx1):
        x1sums_av[i] += x1sums[n][i]

    for j in range(nx2):
        x2sums_av[j] += x2sums[n][j]

    if out_every > 0 and n % out_every == 0:
        print(f"Observables: iteration {n} complete")


assert sum_diffs_mag    >= 0.
assert sum_diffs_energy >= 0.

var_mag    = sum_diffs_mag   /(1.*(ncalc-1)) if ncalc > 1 else 0.
var_energy = sum_diffs_energy/(1.*(ncalc-1)) if ncalc > 1 else 0.

chi = c1*var_mag     # Magnetic susceptibility
cv  = c2*var_energy  # Specific heat

x1sums_av /= ncalc
x2sums_av /= ncalc

print("Done calculating the ensemble-averaged observables")


fout = h5py.File(outfile, "w")

dset_mag_av    = fout.create_dataset("Ensemble-averaged magnetization", data = mag_av)
dset_energy_av = fout.create_dataset("Ensemble-averaged energy",        data = energy_av)
dset_chi       = fout.create_dataset("Magnetic susceptibility",         data = chi)
dset_cv        = fout.create_dataset("Specific heat",                   data = cv)



# Correlation function
if nx1 % 2 == 0: nx1_half = int(nx1/2)
else:            nx1_half = int(nx1/2) + 1

if nx2 % 2 == 0: nx2_half = int(nx2/2)
else:            nx2_half = int(nx2/2) + 1


if nx1 == nx2:
    n_half = nx1_half
    assert nx2_half == n_half

    corr = np.zeros(n_half)

    for n in range(ncalc):
        for dist in range(1, n_half):
            sum_prod = 0

            for i in range(nx1):
                i_shift   = i + dist if i + dist < nx1 else i + dist - nx1
                sum_prod += (x1sums[n][i] - x1sums_av[i])*(x1sums[n][i_shift] - x1sums_av[i_shift]) \
                          + (x2sums[n][i] - x2sums_av[i])*(x2sums[n][i_shift] - x2sums_av[i_shift])

            corr[dist-1] += sum_prod

        if out_every > 0 and n % out_every == 0:
            print(f"Correlation: iteration {n} complete")

    corr     /= (2.*ncalc*nx1)
    dset_corr = fout.create_dataset("Correlation function along rows+columns", data = corr)


else:  # nx1 != nx2
    corr_rows = np.zeros(nx1_half)
    corr_cols = np.zeros(nx2_half)

    for n in range(ncalc):
        for dist1 in range(1, nx1_half):
            sum_prod = 0

            for i in range(nx1):
                i_shift   = i + dist1 if i + dist1 < nx1 else i + dist1 - nx1 
                sum_prod += (x1sums[n][i] - x1sums_av[i])*(x1sums[n][i_shift] - x1sums_av[i_shift])

            corr_rows[dist1-1] += sum_prod

        for dist2 in range(1, nx2_half):
            sum_prod = 0

            for j in range(nx2):
                j_shift   = j + dist2 if j + dist2 < nx2 else j + dist2 - nx2
                sum_prod += (x2sums[n][j] - x2sums_av[j])*(x2sums[n][j_shift] - x2sums_av[j_shift])

            corr_cols[dist2-1] += sum_prod

        if out_every > 0 and n % out_every == 0:
            print(f"Correlation: iteration {n} complete")

    corr_rows /= (1.*ncalc*nx1)
    corr_cols /= (1.*ncalc*nx2)

    dset_rows = fout.create_dataset("Correlation function along rows",    data = corr_rows)
    dset_cols = fout.create_dataset("Correlation function along columns", data = corr_cols)


fout.close()
