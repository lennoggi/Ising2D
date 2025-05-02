#ifndef DECLARE_FUNCTIONS_HH
#define DECLARE_FUNCTIONS_HH

#include <array>
#include <random>
#include <hdf5.h>
#include "Declare_variables.hh"


std::array<int, 7>
set_indices_neighbors_parity(const int &rank,
                             const int &nprocs);

void update(const int                                    &rank,
                  std::mt19937                           &gen,
                  std::uniform_real_distribution<double> &dist,
            const std::array<int, 7>                     &indices_neighbors_parity,
                  std::array<int, nx1locp2_nx2locp2>     &local_lattice);

void write_lattice(const int &rank,
                   const int &nprocs,
                   const int &x1index,
                   const int &x2index,
                   const int &n,
                   const std::array<int, nx1locp2_nx2locp2> &local_lattice,
                   const hid_t &file_id);


#endif  // DECLARE_FUNCTIONS_HH
