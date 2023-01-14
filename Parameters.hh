#ifndef PARAMETERS_HH
#define PARAMETERS_HH


/* beta = 1/(kB*T) (kB = 1 here)
 * Critical value: 0.4406868 (infinite lattice)                                 */
#define BETA 0.42

// Full lattice size
#define Lx 256
#define Ly 256

// Number of ranks along x and y
#define Nx 4
#define Ny 2

// Number of lattice updates to reach thermal equilibrium
#define N_THERM 200000

/* Number of lattice updates in which the expectation values of all the
 * observables are computed after reaching thermal equilibrium                  */
#define N_CALC 2000000


#endif PARAMETERS_HH
