#ifndef TYPES_HH
#define TYPES_HH

// Custom datatype storing the neighbors and parity of an MPI process
typedef struct {
   int  x1_down; 
   int  x1_up; 
   int  x2_down; 
   int  x2_up; 
   bool parity; 
} neighbors_and_parity_t;


#endif  // TYPES_HH
