#include <cassert>
#include <array>
#include <vector>
#include <hdf5.h>

#include "include/Declare_variables.hh"
#include "include/Declare_functions.hh"
#include "include/Macros.hh"

using namespace std;


/* =============================================================
 * Routine writing the full lattice to a dataset in an HDF5 file
 * ============================================================= */
void write_lattice(const int &rank,
                   const int &nprocs,
                   const int &x1index,
                   const int &x2index,
                   const int &n,
                   const array<int, nx1locp2_nx2locp2> &local_lattice,
                   const hid_t &file_id) {
        const array<hsize_t, 2> dims_global = {NY, NX};
        const auto              fspace_id   = H5Screate_simple(2, dims_global.data(), nullptr);

        /* TODO: consider creating a compressed and/or chunked and/or
         *       cache-optimized dataset                                        */
        ostringstream dset_name_ss;
        dset_name_ss << "/Iteration_" << n;
        auto dset_id = H5Dcreate(file_id, dset_name_ss.str().c_str(), H5T_NATIVE_INT, fspace_id,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset_id > 0);


        /* Each process writes its local lattice to file, and HDF5 reshapes the
         * 1D array into a 2D chunk by selecting the hyperslab properly
         * NOTE: think of the full grid as:
         *      |----------------|
         *   x1 |  0  1  2  3  4 |  Example with NPROCS_X1=4, NPROCS_X2=5
         *   |  |  5  6  7  8  9 |
         *   v  | 10 11 12 13 14 |
         *      | 15 16 17 18 19 |
         *      |----------------|
         *                  x2 ->                                               */
        array<hsize_t, 2> offset             = {x1index*nx1loc, x2index*nx2loc};
        array<hsize_t, 2> local_lattice_size = {nx1loc, nx2loc};
        CHECK_ERROR(rank, H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, offset.data(), nullptr, local_lattice_size.data(), nullptr));

        const auto memspace_id = H5Screate_simple(2, local_lattice_size.data(), nullptr);
        assert(memspace_id > 0);

        auto dxpl_id = H5Pcreate(H5P_DATASET_XFER);
        assert(dxpl_id > 0);
        CHECK_ERROR(rank, H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE));

        CHECK_ERROR(rank, H5Dwrite(dset_id, H5T_NATIVE_INT, memspace_id, fspace_id, dxpl_id, local_lattice.data()));

        CHECK_ERROR(rank, H5Pclose(dxpl_id));
        CHECK_ERROR(rank, H5Sclose(memspace_id));
        CHECK_ERROR(rank, H5Dclose(dset_id));
        CHECK_ERROR(rank, H5Sclose(fspace_id));

    return;
}
