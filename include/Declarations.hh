#ifndef DECLARATIONS_HH
#define DECLARATIONS_HH

#include <array>
#include <H5Cpp.h>
#include <mpi.h>
#include "../Parameters.hh"


// 2*beta
constexpr inline double _2beta = 2.*BETA;

// Process-local lattice size -- interior only
constexpr inline int nxloc = NX/NPROCS_X;
constexpr inline int nyloc = NY/NPROCS_Y;

// Indices of the last ghost row and column on the process-local lattice
constexpr inline int nxloc_p1 = nxloc + 1;
constexpr inline int nyloc_p1 = nyloc + 1;

// Process-local lattice size -- including ghosts
constexpr inline int nxloc_p2 = nxloc + 2;
constexpr inline int nyloc_p2 = nyloc + 2;

// Size of the chunks of data to send to and receive from each process
constexpr inline int nxloc_half = nxloc/2;
constexpr inline int nyloc_half = nyloc/2;

// Size of the full interior of the process-local lattice
constexpr inline int nxloc_nyloc = nxloc*nyloc;


/* Function declarations
 * --------------------- */
void check_parameters(const int &nprocs);

void update(const int  &right,
            const int  &left,
            const int  &up,
            const int  &down,
            const bool &parity,
            std::array<std::array<int, nxloc_p2>, nyloc_p2> &local_lattice);

void write_full_lattice(const int &proc_ID,
                        const int &nprocs,
                        const int &n,
                        const std::array<std::array<int, nxloc_p2>, nyloc_p2> &local_lattice,
                        const H5::H5File &outfile);

void thermalize(const int  &nprocs,
                const int  &proc_ID,
                const int  &right,
                const int  &left,
                const int  &up,
                const int  &down,
                const bool &parity,
                std::array<std::array<int, nxloc_p2>, nyloc_p2> &local_lattice);


#endif  // DECLARATIONS_HH
