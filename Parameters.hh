#ifndef PARAMETERS_HH
#define PARAMETERS_HH


/* beta = 1/(kB*T) (kB = 1 here)
 * Critical value: 0.4406868 (infinite lattice)                                 */
#define BETA 0.42

// Full lattice size
#define NX 256
#define NY 256

// Number of ranks along x and y
#define N_PROCS_X 4
#define N_PROCS_Y 4

// Number of lattice updates to reach thermal equilibrium
#define N_THERM 200000

/* Number of lattice updates during which the expectation values of all the
 * observables are computed after reaching thermal equilibrium                  */
#define N_CALC 2000000

// Verbosity
#define VERBOSE     true
#define VERYVERBOSE true


#endif  // PARAMETERS_HH
