#ifndef PARAMETERS_HH
#define PARAMETERS_HH


/* beta = 1/(kB*T) (kB = 1 here)
 * Critical value: 0.5*log(1 + sqrt(2)) ~= 0.4406868 (infinite lattice)                                 */
#define BETA 0.42


/* Number of MPI processes along x and y
 * NOTE: be sure to request NPROCS_X1*NPROCS_X2 MPI processes in your job
 *       submission script                                                      */
#define NPROCS_X1 8
#define NPROCS_X2 8

/* Total number of points along the two lattice dimensions
 * NOTE: both NX1 and NX2 must be EVEN integers                                  */
#define NX1 128
#define NX2 128


/* The lattice update kernel is launched on each quarter of the process-local
 * lattice (see Update_device.cu) using a single thread block of
 * (NX1/NPROCS_X1)*(NX2/NPROCS_X2) threads. However, CUDA limits the number of
 * threads in a block is limited to a maximum of 1024; also, the typical number
 * of threads per block that leads to optimal performance is 128/256 (although
 * the ultimate answer can only be given by timing and/or profiling the
 * application). Therefore, we provide parameters MAX_BLOCK_SIZE_X1 and
 * MAX_BLOCK_SIZE_X2 to limit the number of threads per block.
 *
 * NOTE: CUDA schedules threads to be executed in warps of 32, so having less
 *   than 32 threads per blocks (MAX_BLOCK_SIZE_X1*MAX_BLOCK_SIZE_X2 < 32) is
 *   NOT recommended (and in fact forbidden by include/Check_parameters.hh), as
 *   that would leave some of the warp slots unused.
 *
 * RULE OF THUMB: MAX_BLOCK_SIZE_X1*MAX_BLOCK_SIZE_X2 = 128--256 is generally
 *   optimal, but the ultimate answer can only be given by timing and/or
 *   profiling the application                                                  */
#ifdef USE_CUDA
#define MAX_BLOCK_SIZE_X1 16
#define MAX_BLOCK_SIZE_X2 16
#endif


// Number of lattice updates to reach thermal equilibrium
#define NTHERM 200000

/* Number of lattice updates during which the expectation values of all the
 * observables are computed after reaching thermal equilibrium                  */
#define NCALC 10


// Whether to save lattice data during thermalization
#define SAVE_LATTICE_THERM true

// Whether to save lattice data during the calculation of the observables
#define SAVE_LATTICE_CALC true

// Output frequency
#define OUT_EVERY 200000


// Whether to have one stdout and one stderr file per MPI rank
#define OUTERR_ALL_RANKS true

// Verbosity
#define VERBOSE     true
#define VERYVERBOSE false


#endif  // PARAMETERS_HH
