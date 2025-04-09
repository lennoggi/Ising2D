#ifndef MACROS_HH
#define MACROS_HH

#include <iostream>  // So that std::cout and std::endl work  when the macros below are used
#include <sstream>   // So that 'msg_ss.str()'          works when the macros below are used
#include <mpi.h>     // So that MPI_abort()             works when the macros below are used


// Macro printing an informative message
#define INFO(rankid, msg_ss)                      \
    do {                                          \
        std::ostringstream ss;                    \
        ss << "INFO (rank " << rankid << "): "    \
           << msg_ss << endl;                     \
                                                  \
        fprintf(stdout, "%s", ss.str().c_str());  \
        fflush(stdout);                           \
    } while(0)


// Macro printing a warning message
#define WARNING(rankid, msg_ss)                    \
    do {                                           \
        std::ostringstream ss;                     \
        ss << "WARNING (rank " << rankid << "): "  \
           << msg_ss << endl;                      \
                                                   \
        fprintf(stderr, "%s", ss.str().c_str());   \
        fflush(stderr);                            \
    } while(0)


// Macro printing an error message and aborting program execution
#define ERROR(rankid, msg_ss)                                  \
    do {                                                       \
        std::ostringstream ss;                                 \
        ss << "ERROR (rank " << rankid                         \
           << ", file " << __FILE__ << ", line " << __LINE__   \
           << endl << "  -> " << msg_ss << endl;               \
                                                               \
        fprintf(stderr, "%s", ss.str().c_str());               \
        fflush(stderr);                                        \
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_INTERN);             \
    } while(0)


// Macro checking for errors coming from routines returning error codes
#define CHECK_ERROR(rankid, routine)                    \
do {                                                    \
    const int err = routine;                            \
    if (err < 0) {                                      \
        ERROR(rankid, "Routine '" << #routine <<        \
              "' returned error code " << err << ")");  \
    }                                                   \
} while (0)


#endif  // MACROS_HH
