#ifndef DECLARATIONS_HH
#define DECLARATIONS_HH

#include "../Parameters.hh"


// Process-local interior lattice size
constexpr inline int nx1loc = NX1/NPROCS_X1;
constexpr inline int nx2loc = NX2/NPROCS_X2;

/* Both nx1loc=NX1/NPROCS_X1 and nx2loc=NX2LOC/NPROCS_X2 must be EVEN for the
 * white/black update method to work (see Update.cc)                            */
static_assert(nx1loc % 2 == 0);
static_assert(nx2loc % 2 == 0);

// Index of the last ghost column on the process-local lattice
constexpr inline int nx2loc_p1 = nx2loc + 1;

/* Full extent (including ghost points) of the process-local lattice along x2
 * (needed to build the index along x1 in the flattened process-local lattice)  */
constexpr inline int nx1loc_p2 = nx1loc + 2;
constexpr inline int nx2loc_p2 = nx2loc + 2;

// Total size (including ghost points) of the flattened process-local lattice
constexpr inline int nx1locp2_nx2locp2 = nx1loc_p2*nx2loc_p2;

// Helper variables for communication
constexpr inline int nx1locp2_p1          = nx1loc + 3;
constexpr inline int nx1loc_nx2loc        = nx1loc*nx2loc;
constexpr inline int nx1loc_nx2locp2_p1   = nx1loc_nx2loc + 3;
constexpr inline int nx1locp1_nx2locp2_p1 = nx1loc_nx2loc + 4;


#endif  // DECLARATIONS_HH
