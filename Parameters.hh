#ifndef PARAMETERS_HH
#define PARAMETERS_HH


/* beta = 1/(kB*T) (kB = 1 here)
 * Critical value: 0.5*log(1 + sqrt(2)) ~= 0.4406868 (infinite lattice)                                 */
#define BETA 0.5


/* Number of MPI processes along x and y
 * NOTE: be sure to request NPROCS_X1*NPROCS_X2 MPI processes in your job
 *       submission script                                                      */
#define NPROCS_X1 6
#define NPROCS_X2 8

/* Total number of points along the two lattice dimensions
 * NOTE: both NX1 and NX2 must be EVEN integers                                  */
#define NX1 384
#define NX2 512

/* Number of CUDA threads per block along the two lattice dimensions
 * NOTE: the pseudorandom-number-generation kernel runs on ALL of the interior
 *   points of the process-local lattice, whereas the lattice update kernel only
 *   runs on HALF of them ("red/black" checkerboard logic)
 * NOTE: the number of blocks ('grid size') is determined by the block size and
 *   the size of the process-local lattice
 * NOTE: CUDA schedules threads to be executed in warps of 32, so having less
 *   than 32 threads per blocks (BLOCK_SIZE < 32) is NOT recommended, as it
 *   leaves some of the warp slots unused. On the other hand, the typical
 *   maximum number of threads per block on many GPU architectures is 1024, so
 *   BLOCK_SIZE > 1024 is generally illegal. Also, maximizing the number of
 *   threads per block (i.e., BLOCK_SIZE = 1024) is doesn't always improve
 *   performance; in fact, it may sometimes HURT performance.
 * RULE OF THUMB: BLOCK_SIZE = 128--256 is generally optimal, but the ultimate
 *   answer can only be given by timing and/or profiling the application        */
#ifdef USE_CUDA
#define BLOCK_SIZE_X1 16
#define BLOCK_SIZE_X2 16
#endif


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
