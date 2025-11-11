#ifndef DECLARATIONS_HH
#define DECLARATIONS_HH

#include "../Parameters.hh"


// Process-local interior lattice size
constexpr inline int nx1loc = NX1/NPROCS_X1;
constexpr inline int nx2loc = NX2/NPROCS_X2;

// Indices of the last ghost row and column on the process-local lattice
constexpr inline int nx1loc_p1 = nx1loc + 1;
constexpr inline int nx2loc_p1 = nx2loc + 1;

/* Full extent (including ghost points) of the process-local lattice along x2
 * (needed to build the index along x1 in the flattened process-local lattice)  */
constexpr inline int nx1loc_p2 = nx1loc + 2;
constexpr inline int nx2loc_p2 = nx2loc + 2;

// Helper variable for communication
constexpr inline int nx1loc_nx2loc = nx1loc*nx2loc;

// Total size (including ghost points) of the flattened process-local lattice
constexpr inline int nx1locp2_nx2locp2 = nx1loc_p2*nx2loc_p2;

/* Both nx1loc=NX1/NPROCS_X1 and nx2loc=NX2LOC/NPROCS_X2 must be EVEN for the
 * parity update method to work (see Update.cc and Update_device.cu)
 * Additionally, on GPUs, the update happens in a 'checkerboard' fashion,
 * whereby all 'black' sites are updated first and all 'red' sites are updated
 * next to preserve detailed balance (see Update_device.cu)                     */
static_assert(nx1loc % 2 == 0);
static_assert(nx2loc % 2 == 0);
constexpr inline int nx1loc_div2 = nx1loc/2;
constexpr inline int nx2loc_div2 = nx2loc/2;

#ifdef USE_CUDA
/* On GPUs, lattice sites are updated in a 'checkerboard' fashion, whereby all
 * 'black' sites are updated first and all 'red' sites are updated next to
 * preserve detailed balance (see Update_device.cu)                             */
static_assert(nx2loc % 4 == 0);
constexpr inline int nx2loc_div4 = nx2loc/4;
#endif


#endif  // DECLARATIONS_HH
