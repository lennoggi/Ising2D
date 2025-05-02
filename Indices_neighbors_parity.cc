#include <cstdlib>
#include <cassert>
#include <array>

#include "Parameters.hh"

using namespace std;


/* ============================================================================
 * Routine setting the process' indices along x1 and x2, its neighbors, and its
 * parity
 * ============================================================================
 *
 * Think of the full grid e.g. as:
 *        |----------------|
 *   x1 0 | 0  1  2  3  4  |  Example with NPROCS_X1=4, NPROCS_X2=5
 *   |  1 | 5  6  7  8  9  |
 *   v  2 | 10 11 12 13 14 |
 *      3 | 15 16 17 18 19 |
 *        |----------------|
 *          0  1  2  3  4
 *                  x2 ->
 * with both x1 and x2 being periodic (torus topology)
 * TODO: implement different domain topologies with different boundary conditions:
 *   - Periodic (already done)
 *   - Hard boundary on e.g. x with fixed +1 or -1 boundary points
 *   - (...)
 */
array<int, 7>
set_indices_neighbors_parity(const int &rank,
                             const int &nprocs) {
    const auto indices = div(rank, NPROCS_X2);
    const auto x1index = indices.quot;
    const auto x2index = indices.rem;

    assert(x1index >= 0 and x1index < NPROCS_X1);
    assert(x2index >= 0 and x2index < NPROCS_X2);


    constexpr int diff_rank_x1 = NPROCS_X2*(NPROCS_X1 - 1);

    const auto x1down = (x1index == NPROCS_X1 - 1) ? rank - diff_rank_x1 : rank + NPROCS_X2;
    const auto x1up   = (x1index == 0)             ? rank + diff_rank_x1 : rank - NPROCS_X2;

    assert(x1down >= 0 and x1down < nprocs);
    assert(x1up   >= 0 and x1up   < nprocs);

    const auto x2down = (x2index == 0)             ? rank + NPROCS_X2 - 1 : rank - 1;
    const auto x2up   = (x2index == NPROCS_X2 - 1) ? rank - NPROCS_X2 + 1 : rank + 1;

    assert(x2down >= 0 and x2down < nprocs);
    assert(x2up   >= 0 and x2up   < nprocs);


    // This relies on having an EVEN number of processes along x1 and x2
    const bool parity = ((x1index + rank) & 1);

    // Sanity check: the full lattice should look like a 'chessboard'
    const array<int, 4> neighbors = {x1down, x1up, x2down, x2up};

    for (const auto &nb : neighbors) {
        const int  x1index_nb = nb/NPROCS_X2;
        const bool parity_nb  = ((x1index_nb + nb) & 1);
        assert(parity_nb != parity);
    }

    return array<int, 7> {x1index, x2index, x1down, x1up, x2down, x2up, parity};
}
