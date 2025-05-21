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


# Objects to be built
OBJ = Indices_neighbors.o Main.o Update.o Write_lattice.o
EXE = Ising2D_exe

ifneq (,$(findstring -DUSE_CUDA,$(CXXFLAGS)))
OBJ += CUDA_utils.o
endif


# Build all targets
build: $(OBJ)
	$(CXX) -o $(EXE) $(OBJ) ${LDFLAGS} -L$(HDF5_LIBS_DIR) $(HDF5_LIBS) -L$(CUDA_LIBS_DIR) $(CUDA_LIBS)

Indices_neighbors.o: Indices_neighbors.cc Parameters.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -c Indices_neighbors.cc

Main.o: Main.cc include/Check_parameters.hh include/Declare_variables.hh include/Declare_functions.hh include/Macros.hh Parameters.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -I$(HDF5_INC_DIR) -c Main.cc

Write_lattice.o: Write_lattice.cc include/Declare_variables.hh include/Declare_functions.hh include/Macros.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -I$(HDF5_INC_DIR) -c Write_lattice.cc

Update.o: Update.cc include/Declare_variables.hh include/Declare_functions.hh include/Macros.hh Parameters.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -I$(HDF5_INC_DIR) -c Update.cc

ifneq (,$(findstring -DUSE_CUDA,$(CXXFLAGS)))
CUDA_utils.o: CUDA_utils.cu include/Declare_functions.hh include/Macros.hh
	$(CUCC) -c CUDA_utils.cu $(CUCC_FLAGS)
endif



# Remove the executable and all object files
# NOTE: on make clean, the optionlist is not parsed and thus CXXFLAGS remains
#   empty. Therefore, -DUSE_CUDA is not found in CXXFLAGS and CUDA_utils.o is
#   not added to $(OBJ), so CUDA_utils.o must be removed explicitly.
# NOTE: icpx generates *.o.tmp files which should be removed when cleaning
.PHONY : clean
clean:
	${RM} ${EXE} ${OBJ} CUDA_utils.o *.o.tmp
