# Include and parse the option list, but don't do that when running 'make clean'
ifeq (, ${filter clean, ${MAKECMDGOALS}})
	include ${options}

	# Disable optimization if so requested by the user
	ifeq (${OPTIMIZE}, no)
		CXX_OPTIMIZE_FLAGS =
	else ifneq (${OPTIMIZE}, yes)
		$(error Set option OPTIMIZE to either yes or no in the optionlist (current setting: ${OPTIMIZE}))
	endif

	# Disable warnings if so requested by the user
	ifeq (${WARN}, no)
		CXX_WARN_FLAGS =
	else ifneq (${WARN}, yes)
		$(error Set option WARN to either yes or no in the optionlist (current setting: ${WARN}))
	endif

	# Disable debug mode if so requested by the user
	ifeq (${DEBUG}, no)
		CXX_DEBUG_FLAGS =
	else ifneq (${DEBUG}, yes)
		$(error Set option DEBUG to either yes or no in the optionlist (current setting: ${DEBUG}))
	endif
endif


# Objects to be built from regular C++ (not CUDA) source files
OBJ = Indices_neighbors_parity.o Main.o Observables_correlation.o Update.o Write_lattice.o
EXE = Ising2D_exe

ifneq (,$(findstring -DUSE_CUDA,$(CXXFLAGS)))
OBJ += Random_numbers.o Utils.o Update_device.o
endif


# Build all targets
build: $(OBJ)
ifneq (,$(findstring -DUSE_CUDA,$(CXXFLAGS)))
	$(CXX) -o $(EXE) $(OBJ) ${LDFLAGS} -L$(HDF5_LIBS_DIR) $(HDF5_LIBS) -L$(CUDA_LIBS_DIR) $(CUDA_LIBS)
else
	$(CXX) -o $(EXE) $(OBJ) ${LDFLAGS} -L$(HDF5_LIBS_DIR) $(HDF5_LIBS)
endif

Indices_neighbors_parity.o: Indices_neighbors_parity.cc Parameters.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -c Indices_neighbors_parity.cc

Main.o: Main.cc include/Check_parameters.hh include/Declare_variables.hh include/Declare_functions.hh include/Macros.hh Parameters.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -I$(HDF5_INC_DIR) -c Main.cc

Observables_correlation.o: Observables_correlation.cc include/Declare_variables.hh include/Macros.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -I$(HDF5_INC_DIR) -c Observables_correlation.cc

Write_lattice.o: Write_lattice.cc include/Declare_variables.hh include/Declare_functions.hh include/Macros.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -I$(HDF5_INC_DIR) -c Write_lattice.cc

Update.o: Update.cc include/Declare_variables.hh include/Declare_functions.hh include/Macros.hh Parameters.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -I$(HDF5_INC_DIR) -c Update.cc

ifneq (,$(findstring -DUSE_CUDA,$(CXXFLAGS)))
Random_numbers.o: Random_numbers.cu include/Declare_functions.hh include/Macros.hh
	$(CUCC) $(CUCC_FLAGS) -I$(HDF5_INC_DIR) -c Random_numbers.cu

Utils.o: Utils.cu include/Declare_functions.hh include/Macros.hh
	$(CUCC) $(CUCC_FLAGS) -I$(HDF5_INC_DIR) -c Utils.cu

Update_device.o: Update_device.cu include/Declare_variables.hh include/Declare_functions.hh include/Macros.hh Parameters.hh
	$(CUCC) $(CUCC_FLAGS) -I$(HDF5_INC_DIR) -c Update_device.cu
endif



# Remove the executable and all object files
# NOTE: on make clean, the optionlist is not parsed and thus CXXFLAGS remains
#   empty. Therefore, -DUSE_CUDA is not found in CXXFLAGS and the object files
#   generated from .cu files are not added to $(OBJ), so those object files must
#   be removed explicitly.
# NOTE: icpx generates *.o.tmp files which should be removed when cleaning
.PHONY : clean
clean:
	${RM} ${EXE} ${OBJ} Random_numbers.o Utils.o Update_device.o *.o.tmp
