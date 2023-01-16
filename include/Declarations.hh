#ifndef DECLARATIONS_HH
#define DECLARATIONS_HH

#include <array>
#include <mpi.h>
#include "../Parameters.hh"


// 2*beta
constexpr inline double _2beta = 2.*BETA;

// Process-local lattice size -- interior only
constexpr inline int nx_local = NX/N_PROCS_X;
constexpr inline int ny_local = NX/N_PROCS_X;

// Indices of the last ghost row and column on the process-local lattice
constexpr inline int nx_local_p1 = nx_local + 1;
constexpr inline int ny_local_p1 = ny_local + 1;

// Process-local lattice size -- including ghosts
constexpr inline int nx_local_p2 = nx_local + 2;
constexpr inline int ny_local_p2 = ny_local + 2;

// Size of the chunks of data to send to and receive from each process
constexpr inline int nx_local_half = nx_local/2;
constexpr inline int ny_local_half = ny_local/2;


// Function declarations
void check_parameters(const int &N_procs);

void update(const int        &right,
            const int        &left,
            const int        &up,
            const int        &down,
            const bool       &parity,
            std::array<std::array<int, ny_local_p2>, nx_local_p2> &lattice);


#endif  // DECLARATIONS_HH
