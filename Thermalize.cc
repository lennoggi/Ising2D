#include <array>
#include <string>
#include <sstream>
#include <H5Cpp.h>

#include "include/Declarations.hh"
#include "include/Macros.hh"
#include "Parameters.hh"

using namespace std;
using namespace H5;


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
    #if (SAVE_LATTICE_DURING_THERMALIZATION)
        #if (VERBOSE)
            INFO("Saving the lattice to HDF5 files during thermalization");
        #endif

        ostringstream filename_ss;
        filename_ss << "Full_lattice_proc_" << proc_ID << ".h5";
        H5File outfile(filename_ss.str(), H5F_ACC_TRUNC);
    #endif


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
