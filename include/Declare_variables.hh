#ifndef DECLARATIONS_HH
#define DECLARATIONS_HH

#include "../Parameters.hh"


// Process-local interior lattice size
constexpr inline int nx1loc = NX/NPROCS_X1;
constexpr inline int nx2loc = NY/NPROCS_X2;

/* NOTE: both nx1loc=NX/NPROCS_X1 and nx2loc=NYLOC/NPROCS_X2 must be EVEN in
 *   order for communication among MPI processes to happen smoothly using the
 *   'parity' technique                                                         */
static_assert(nx1loc % 2 == 0);
static_assert(nx2loc % 2 == 0);

// Indices of the last ghost row and column on the process-local lattice
//constexpr inline int nx1loc_p1 = nx1loc + 1;
//constexpr inline int nx2loc_p1 = nx2loc + 1;

// Process-local total (i.e., including ghosts) lattice size
//constexpr inline int nx1loc_p2 = nx1loc + 2;
//constexpr inline int nx2loc_p2 = nx2loc + 2;

// Size of the chunks of data to send to and receive from each process
//constexpr inline int nx1loc_half = nx1loc/2;
//constexpr inline int nx2loc_half = nx2loc/2;

// Size of the process-local lattice
//constexpr inline int nx1loc_nx2loc = nx1loc*nx2loc;
constexpr inline int nx1locp2_nx2locp2 = (nx1loc + 2)*(nx2loc + 2);


#endif  // DECLARATIONS_HH
