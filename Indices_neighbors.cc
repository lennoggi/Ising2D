#include <cstdlib>
#include <cassert>
#include <array>

#include "Parameters.hh"

using namespace std;


/* ======================================================================
 * Routine setting the process' indices along x1 and x2 and its neighbors
 * ======================================================================
 *
 * Think of the full grid e.g. as:
 *        |----------------|
 *   ^  3 | 15 16 17 18 19 |  Example with NPROCS_X1=4, NPROCS_X2=5
 *   |  2 | 10 11 12 13 14 |
 *   x1 1 | 5  6  7  8  9  |
 *      0 | 0  1  2  3  4  |
 *        |----------------|
 *          0  1  2  3  4
 *                  x2 ->
 * with both x1 and x2 being periodic (torus topology)
 * TODO: implement different domain topologies with different boundary conditions:
 *   - Periodic (already done)
 *   - Hard boundary on e.g. x with fixed +1 or -1 boundary points
 *   - (...)
 */
array<int, 6>
set_indices_neighbors(const int &rank,
                      const int &nprocs) {
    const auto indices = div(rank, NPROCS_X2);
    const auto x1index = indices.quot;
    const auto x2index = indices.rem;

    assert(x1index >= 0 and x1index < NPROCS_X1);
    assert(x2index >= 0 and x2index < NPROCS_X2);

    constexpr int diff_rank_x1 = NPROCS_X2*(NPROCS_X1 - 1);

    const auto x1down = (x1index == 0)             ? rank + diff_rank_x1 : rank - NPROCS_X2;
    const auto x1up   = (x1index == NPROCS_X1 - 1) ? rank - diff_rank_x1 : rank + NPROCS_X2;

    assert(x1down >= 0 and x1down < nprocs);
    assert(x1up   >= 0 and x1up   < nprocs);

    const auto x2down = (x2index == 0)             ? rank + NPROCS_X2 - 1 : rank - 1;
    const auto x2up   = (x2index == NPROCS_X2 - 1) ? rank - NPROCS_X2 + 1 : rank + 1;

    assert(x2down >= 0 and x2down < nprocs);
    assert(x2up   >= 0 and x2up   < nprocs);

    return array<int, 6> {x1index, x2index, x1down, x1up, x2down, x2up};
}
