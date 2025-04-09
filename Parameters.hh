#ifndef PARAMETERS_HH
#define PARAMETERS_HH


/* beta = 1/(kB*T) (kB = 1 here)
 * Critical value: 0.4406868 (infinite lattice)                                 */
#define BETA 0.42


/* Number of MPI processes along x and y
 * NOTE: be sure to request NPROCS_X1*NPROCS_X2 MPI processes in your job
 *       submission script                                                      */
#define NPROCS_X1 4
#define NPROCS_X2 4

/* Total number of points along the two lattice directions
 * NOTE: both NX and BY must be EVEN integers                                   */
#define NX 128
#define NY 128


// Number of lattice updates to reach thermal equilibrium
#define NTHERM 200000

/* Number of lattice updates during which the expectation values of all the
 * observables are computed after reaching thermal equilibrium                  */
#define NCALC 2000000


// Whether to save lattice data during thermalization
#define SAVE_LATTICE_THERM true

// Whether to save lattice data during the calculation of the observables
#define SAVE_LATTICE_CALC true

/* Whether to output a single file or one file per process
 * NOTE: outputting one file per process is likely slower, but reduces the
 *   stress on the filesystem and there is no need to recombine data from
 *   different processes after the simulation                                   */
#define SINGLE_OUTPUT_FILE true

// Output frequency
#define OUT_EVERY 1000


// Whether to have one stdout and one stderr file per MPI rank
#define OUTERR_ALL_RANKS true

// Verbosity
#define VERBOSE     true
//#define VERYVERBOSE true


#endif  // PARAMETERS_HH
