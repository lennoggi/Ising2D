#ifndef PARAMETERS_HH
#define PARAMETERS_HH


/* beta = 1/(kB*T) (kB = 1 here)
 * Critical value: 0.4406868 (infinite lattice)                                 */
#define BETA 0.42

// Full lattice size
#define NX 256
#define NY 256

// Number of ranks along x and y
#define NPROCS_X 4
#define NPROCS_Y 4

// Number of lattice updates to reach thermal equilibrium
#define NTHERM 200000

/* Number of lattice updates during which the expectation values of all the
 * observables are computed after reaching thermal equilibrium                  */
#define NCALC 2000000

// Verbosity
#define VERBOSE     true
#define VERYVERBOSE true

/* Whether to save the full lattice to a HDF5 file at all timesteps during
 * thermalization or not                                                        */
#define SAVE_LATTICE_DURING_THERMALIZATION true


#endif  // PARAMETERS_HH
