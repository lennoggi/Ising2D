#ifndef PARAMETERS_HH
#define PARAMETERS_HH


/* beta = 1/(kB*T) (kB = 1 here)
 * Critical value: 0.5*log(1 + sqrt(2)) ~= 0.4406868 (infinite lattice)                                 */
#define BETA 0.5


/* Number of MPI processes along x and y
 * NOTE: be sure to request NPROCS_X1*NPROCS_X2 MPI processes in your job
 *       submission script                                                      */
#define NPROCS_X1 8
#define NPROCS_X2 8

/* Total number of points along the two lattice directions
 * NOTE: both NX1 and NX2 must be EVEN integers                                  */
#define NX1 512
#define NX2 512


// Number of lattice updates to reach thermal equilibrium
#define NTHERM 1000

/* Number of lattice updates during which the expectation values of all the
 * observables are computed after reaching thermal equilibrium                  */
#define NCALC 10


// Whether to save lattice data during thermalization
#define SAVE_LATTICE_THERM true

// Whether to save lattice data during the calculation of the observables
#define SAVE_LATTICE_CALC true

// Output frequency
#define OUT_EVERY 1


// Whether to have one stdout and one stderr file per MPI rank
#define OUTERR_ALL_RANKS true

// Verbosity
#define VERBOSE     true
//#define VERYVERBOSE true


#endif  // PARAMETERS_HH
