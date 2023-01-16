#include <cstdlib>
#include <cmath>
#include <array>

#include "include/Declarations.hh"
#include "Parameters.hh"

using namespace std;


/* =============================================================================
 * Routine updating the process-local lattice by dividing the latter in four
 * equal parts and performing communications on each of them one after the other
 * =============================================================================*/
void update(const int        &right,
            const int        &left,
            const int        &up,
            const int        &down,
            const bool       &parity,
            array<array<int, ny_local_p2>, nx_local_p2> &lattice) {

    /* Arrays encoding which processes rows (i.e. x data chunks) and columns
     * (i.e. y data chunks) should be sent to and received from. The
     * process-local lattice is assumed to be splitted in four parts like this:
     *   1  3
     *   0  2                                                                   */
    const array<int, 4> sendrow = {down,  up,    down,  up};
    const array<int, 4> sendcol = {left,  left,  right, right};
    const array<int, 4> recvrow = {up,    down,  up,    down};
    const array<int, 4> recvcol = {right, right, left,  left};

    // Communication tags
    const int tag1 = 1;
    const int tag2 = 2;
    const int tag3 = 3;
    const int tag4 = 4;


    // Loop over the four parts of the process-local lattice
    int count = 0;

    for (int kx = 0; kx < 2; ++kx) {
        const int offset_x = kx*nx_local_half;
        const int imin     = offset_x + 1;
        const int imax     = imin + nx_local_half;

        for (int ky = 0; ky < 2; ++ky) {
            const int offset_y  = ky*ny_local_half;
            const int jmin      = offset_y + 1;
            const int jmax      = jmin + ny_local_half;

            // Update the current part of the process-local lattice
            for (int i = imin; i < imax; ++i) {
                for (int j = jmin; j < jmax; ++j) {
                    const int f = -(lattice.at(i + 1).at(j) +
                                    lattice.at(i - 1).at(j) +
                                    lattice.at(i).at(j + 1) +
                                    lattice.at(i).at(j - 1));
                    const double trial = rand()/(1.*RAND_MAX);
                    const double prob  = 1./(1. + exp(_2beta*f));

                    if (trial < prob) {
                        lattice.at(i).at(j) = 1;
                    }

                    else {
                        lattice.at(i).at(j) = -1;
                    }
                }
            }

            // Set up the chunks of data to be sent out
            const int i_send = (ky == 0) ? 1 : ny_local;
            const int j_send = (kx == 0) ? 1 : nx_local;

            array<int, nx_local_half> outrow, inrow;
            array<int, ny_local_half> outcol, incol;

            for (int j = 1; j <= nx_local_half; ++j) {
                outrow.at(j - 1) = lattice.at(i_send).at(j + offset_x);
            }

            for (int i = 1; i <= ny_local_half; ++i) {
                outcol.at(i - 1) = lattice.at(i + offset_y).at(j_send);
            }


            // Communicate
            if (parity) {
                MPI_Send(outrow.data(), nx_local_half, MPI_INT,
                         sendrow.at(count), tag1, MPI_COMM_WORLD);
                MPI_Recv(inrow.data(),  nx_local_half, MPI_INT,
                         recvrow.at(count), tag2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(outcol.data(), ny_local_half, MPI_INT,
                         sendcol.at(count), tag3, MPI_COMM_WORLD);
                MPI_Recv(incol.data(),  ny_local_half, MPI_INT,
                         recvcol.at(count), tag4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            else {
                MPI_Send(outrow.data(), nx_local_half, MPI_INT,
                         sendrow.at(count), tag2, MPI_COMM_WORLD);
                MPI_Recv(inrow.data(),  nx_local_half, MPI_INT,
                         recvrow.at(count), tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(outcol.data(), ny_local_half, MPI_INT,
                         sendcol.at(count), tag4, MPI_COMM_WORLD);
                MPI_Recv(incol.data(),  ny_local_half, MPI_INT,
                         recvcol.at(count), tag3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }


            // Store the chunks of data received into the ghost rows and columns
            const int i_recv = (ky == 0) ? ny_local_p1 : 0;
            const int j_recv = (kx == 0) ? nx_local_p1 : 0;

            for (int j = 1; j <= nx_local_half; ++j) {
                lattice.at(i_recv).at(j + offset_x) = inrow.at(j - 1);
            }

            for (int i = 1; i <= ny_local_half; ++i) {
                lattice.at(i + offset_y).at(j_recv) = incol.at(i - 1);
            }
        }

        ++count;
    }

    return;
} 
