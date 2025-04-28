#ifndef DECLARE_FUNCTIONS_HH
#define DECLARE_FUNCTIONS_HH

#include <array>
#include <hdf5.h>
#include "Declare_variables.hh"


/* =========================================
 * Routines from Indices_neighbors_parity.cc
 * ========================================= */
std::array<int, 7>
set_indices_neighbors_parity(const int &rank,
                             const int &nprocs);

/* ==============================
 * Routines from Write_lattice.cc
 * ============================== */
void write_lattice(const int &rank,
                   const int &nprocs,
                   const int &x1index,
                   const int &x2index,
                   const int &n,
                   const std::array<int, nx1locp2_nx2locp2> &local_lattice,
                   const hid_t &file_id);

//void update(const int  &left,
//            const int  &right,
//            const int  &up,
//            const int  &down,
//            const bool &parity,
//            std::array<std::array<int, nxloc_p2>, nyloc_p2> &local_lattice);
//
//#if (SINGLE_OUTPUT_FILE)
//void write_full_lattice(const int &proc_ID,
//                        const int &nprocs,
//                        const int &n,
//                        const std::array<std::array<int, nxloc_p2>, nyloc_p2> &local_lattice,
//                        const H5::H5File &outfile);
//#else
//void write_local_lattice(const int &n,
//                         const std::array<std::array<int, nxloc_p2>, nyloc_p2> &local_lattice,
//                         H5::H5File &outfile);
//#endif
//
//void thermalize(const int  &nprocs,
//                const int  &proc_ID,
//                const int  &left,
//                const int  &right,
//                const int  &up,
//                const int  &down,
//                const bool &parity,
//                std::array<std::array<int, nxloc_p2>, nyloc_p2> &local_lattice);


#endif  // DECLARE_FUNCTIONS_HH
