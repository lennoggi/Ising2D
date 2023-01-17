#include <array>
#include <string>
#include <sstream>
#include <H5Cpp.h>

/* FIXME: these are only needed if SINGLE_OUTPUT_FILE is true, but they are not
 *        included if '#if (SINGLE_OUTPUT_FILE) ... #endif' is there            */
#include <cassert>
#include <vector>

#include "include/Declarations.hh"
#include "include/Macros.hh"
#include "Parameters.hh"

using namespace std;
using namespace H5;



/* ##############################
 * #   SINGLE OUTPUT FILE      ##
 * ############################## */

#if (SINGLE_OUTPUT_FILE)

/* =============================================================================
 * Routine writing the full lattice to a dataset in an HDF5 file
 * =============================================================================*/
void write_full_lattice(const int &proc_ID,
		        const int &nprocs,
		        const int &it,
                        const array<array<int, nxloc_p2>, nyloc_p2> &local_lattice,
                        const H5File &outfile) {
    // Send and receive buffers
    const int nprocs_nxloc_nyloc = nprocs*nxloc_nyloc;
    array<int, nxloc_nyloc> sendbuf;
    vector<int> recvbuf(nprocs_nxloc_nyloc);  // nprocs is only known at runtime

    // Fill the send buffer with the interior of the process-local lattice
    int count_sendbuf = 0;
    for (int i = 1; i <= nyloc; ++i) {
        for (int j = 1; j <= nxloc; ++j) {
            sendbuf.at(count_sendbuf) = local_lattice.at(i).at(j);
            ++count_sendbuf;
        }
    }

    // Sanity check
    assert(count_sendbuf == nxloc_nyloc);

    // Gather all process-local lattice interiors at process 0
    MPI_Gather(sendbuf.data(), nxloc_nyloc, MPI_INT,
               recvbuf.data(), nxloc_nyloc, MPI_INT,
               0, MPI_COMM_WORLD);


    if (proc_ID == 0) {
        // Fill in a 2D array representing the interior of the full lattice
        array<array<int, NX>, NY> full_lattice;
        int count_recvbuf = 0;

        for (int py = 0; py < NPROCS_Y; ++py) {
            const int offset_y = py*nyloc;
            for (int px = 0; px < NPROCS_X; ++px) {
                const int offset_x = px*nxloc;
                for (int i = 0; i < nyloc; ++i) {
                    for (int j = 0; j < nxloc; ++j) {
                        full_lattice.at(i + offset_y).at(j + offset_x) =
                            recvbuf.at(count_recvbuf);
                        ++count_recvbuf;
                    }
                }
            }
        }

        // Sanity check
        assert(count_recvbuf == nprocs_nxloc_nyloc);

        try {
            // Create a 2D dataspace
            const hsize_t   dims[2] = {NY, NX};
            const DataSpace dspace(2, dims);

            // Create a 2D dataset with the above dataspace
            ostringstream dset_name_ss;
            dset_name_ss << "Iteration_" << it;
            const auto dset = outfile.createDataSet(dset_name_ss.str(),
                                                    PredType::NATIVE_INT,
                                                    dspace);
            // Write data to the above dataset
            dset.write(full_lattice.data(), PredType::NATIVE_INT);
        }

        catch (const DataSetIException &e) {
            ERROR(e.getDetailMsg());
        }

        catch (const DataSpaceIException &e) {
            ERROR(e.getDetailMsg());
        }
    }

    return;
}





/* =============================================================================
 * Routine thermalizing the process-local lattice and writing the full lattice
 * to a dataset in an HDF5 file using write_full_lattice()
 * =============================================================================*/
void thermalize(const int  &nprocs,
                const int  &proc_ID,
                const int  &right,
                const int  &left,
                const int  &up,
                const int  &down,
                const bool &parity,
                array<array<int, nxloc_p2>, nyloc_p2> &local_lattice) {
    H5File *outfile;

    if (proc_ID == 0) {
        try {
            outfile = new H5File("Full_lattice.h5", H5F_ACC_TRUNC);
        }

        catch (const FileIException &e) {
            ERROR(e.getDetailMsg());
        }
    }


    for (int it = 0; it < NTHERM; ++it) {
        // Save the lattice as it currently is before updating it
        #if (SAVE_LATTICE_DURING_THERMALIZATION)
            if (it % OUT_EVERY == 0) {
                // Write the full lattice to file
                write_full_lattice(proc_ID, nprocs, it, local_lattice, *outfile);

                #if (VERYVERBOSE)
                    INFO("Full lattice written to file by process 0 (iteration "
                             << it << ")");
                #endif
            }
        #endif

        update(right, left, up, down, parity, local_lattice);
    }


    // Save the full lattice after thermalization
    write_full_lattice(proc_ID, nprocs, NTHERM, local_lattice, *outfile);

    if (proc_ID == 0) {
        delete outfile;
        #if (VERBOSE)
            INFO("Full lattice written to file by process 0 after thermalization");
        #endif
    }

    return;
}





/* ##################################
 * #  ONE OUTPUT FILE PER PROCESS  ##
 * ################################## */

#else

/* =============================================================================
 * Routine writing the process-local lattice to a dataset in an HDF5 file
 * =============================================================================*/
// TODO: select the interior of the process-local lattice and only write that to file
void write_local_lattice(const int &n,
                         const array<array<int, nxloc_p2>, nyloc_p2> &local_lattice,
                         H5File &outfile) {
    try {
        // Create a 2D dataspace
        const hsize_t   dims[2] = {nyloc_p2, nxloc_p2};
        const DataSpace dspace(2, dims);

        // Create a 2D dataset with the above dataspace
        ostringstream dset_name_ss;
       	dset_name_ss << "Iteration_" << n;
        const auto dset = outfile.createDataSet(dset_name_ss.str(),
                                                PredType::NATIVE_INT, dspace);
        // Write data to the above dataset
        dset.write(local_lattice.data(), PredType::NATIVE_INT);
    }

    catch (const DataSetIException &e) {
        ERROR(e.getDetailMsg());
    }

    catch (const DataSpaceIException &e) {
        ERROR(e.getDetailMsg());
    }

    return;
}





/* =============================================================================
 * Routine thermalizing the process-local lattice and writing it to a dataset in
 * an HDF5 file (one file per process)  using write_full_lattice()
 * =============================================================================*/
// TODO: select the interior of the process-local lattice and only write that to file
void thermalize(const int  &nprocs,
                const int  &proc_ID,
                const int  &right,
                const int  &left,
                const int  &up,
                const int  &down,
                const bool &parity,
                array<array<int, nxloc_p2>, nyloc_p2> &local_lattice) {

    ostringstream filename_ss;
    filename_ss << "Local_lattice_proc_" << proc_ID << ".h5";
    H5File outfile(filename_ss.str(), H5F_ACC_TRUNC);


    for (int n = 0; n < NTHERM; ++n) {
        // Save the lattice as it currently is before updating it
        #if (SAVE_LATTICE_DURING_THERMALIZATION)
            if (n % OUT_EVERY == 0) {
                // Write the process-local lattice to file
                write_local_lattice(n, local_lattice, outfile);

                #if (VERYVERBOSE)
                    INFO("Local lattice written to file by process 0 (iteration "
                         << n << ")");
                #endif
            }
        #endif

        // Update the process-local lattice
        update(right, left, up, down, parity, local_lattice);
    }


    // Save the process-local lattice after thermalization
    write_local_lattice(NTHERM, local_lattice, outfile);

    #if (VERBOSE)
        INFO("Local lattice written to file by process 0 after thermalization");
    #endif

    return;
}

#endif  // #else (i.e. one output file per process)
