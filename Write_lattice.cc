#include <cassert>
#include <array>
#include <vector>
// XXX XXX XXX XXX XXX XXX
// XXX XXX XXX XXX XXX XXX
// XXX XXX XXX XXX XXX XXX
//#include <iostream>
////#include <iomanip>
// XXX XXX XXX XXX XXX XXX
// XXX XXX XXX XXX XXX XXX
// XXX XXX XXX XXX XXX XXX

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
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    /* Debug-only printout
     * Suggestion: have each process initialize its local lattice with its rank
     *   and check that, after one iteration, the ghosts have been exchanged
     *   as expected                                                            */
    //for (auto i = decltype(nx1loc_p2){0}; i < nx1loc_p2; ++i) {
    //    const auto i_idx = i*nx2loc_p2;
    //    for (auto j = decltype(nx2loc_p2){0}; j < nx2loc_p2; ++j) {
    //        cout << local_lattice.at(i_idx + j) << "\t";
    //        //cout << setw(3) << setfill('0') << local_lattice.at(i_idx + j) << "\t";
    //    }
    //    cout << endl;
    //}
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX

    const array<hsize_t, 2> dims_global = {NX1, NX2};
    const auto fspace_id = H5Screate_simple(2, dims_global.data(), nullptr);
    assert(fspace_id > 0);

    /* TODO: consider creating a compressed and/or chunked and/or
     *       cache-optimized dataset                                            */
    ostringstream dset_name_ss;
    dset_name_ss << "/Iteration_" << n;
    auto dset_id = H5Dcreate(file_id, dset_name_ss.str().c_str(), H5T_NATIVE_INT, fspace_id,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset_id > 0);


    /* Create a memory dataspace representing the process-local lattice, which
     * has ghosts                                                               */
    const array<hsize_t, 2> local_lattice_size_full = {nx1loc_p2, nx2loc_p2};  // Ghosts included
    const auto memspace_id = H5Screate_simple(2, local_lattice_size_full.data(), nullptr);
    assert(memspace_id > 0);

    // Select the interior hyperslab from the process-local lattice
    const array<hsize_t, 2> mem_start = {1, 1};
    const array<hsize_t, 2> mem_count = {nx1loc, nx2loc};

    CHECK_ERROR(rank, H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET,
        mem_start.data(),
        nullptr,    // Null stride = {1, 1}, i.e. elements are selected contiguously (no "jumps") along each dimension
        mem_count.data(),
        nullptr));  // Null block size = {1, 1}, i.e. blocks of one element (i.e. a single element) are selected along each direction


    /* Select the target hyperslab (i.e. the one corresponding to the
     * current process) in the global lattice
     * NOTE: think of the full grid e.g. as:
     *        |----------------|
     *   ^  3 | 15 16 17 18 19 |  Example with NPROCS_X1=4, NPROCS_X2=5
     *   |  2 | 10 11 12 13 14 |
     *   x1 1 | 5  6  7  8  9  |
     *      0 | 0  1  2  3  4  |
     *        |----------------|
     *          0  1  2  3  4
     *                  x2 ->
     * with both x1 and x2 being periodic (torus topology)
     * NOTE: because in the HDF5 file row indices increase from top to bottom,
     *   while x1index increases in the reverse direction, the row offset must
     *   be flipped                                                             */
    const array<hsize_t, 2> offset = {
        static_cast<hsize_t>((NPROCS_X1 - x1index - 1)*nx1loc),
        static_cast<hsize_t>(x2index*nx2loc)
    };  // The global lattice in the file has no ghosts and the offset reflects this

    const auto &local_lattice_size_interior = mem_count;  // {nx1loc, nx2loc}
    CHECK_ERROR(rank, H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, offset.data(), nullptr, local_lattice_size_interior.data(), nullptr));


    // Write the data to file collectively
    auto dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    assert(dxpl_id > 0);
    CHECK_ERROR(rank, H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE));

    CHECK_ERROR(rank, H5Dwrite(dset_id, H5T_NATIVE_INT, memspace_id, fspace_id, dxpl_id,
        local_lattice.data()));  // Full process-local lattice (i.e. with ghosts) passed in, but only the interior was selected for writing

    CHECK_ERROR(rank, H5Sclose(memspace_id));
    CHECK_ERROR(rank, H5Pclose(dxpl_id));
    CHECK_ERROR(rank, H5Dclose(dset_id));
    CHECK_ERROR(rank, H5Sclose(fspace_id));

    return;
}
