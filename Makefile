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


# Build all targets
build: $(OBJ)
	$(CXX) -o $(EXE) $(OBJ) ${LDFLAGS} -L$(HDF5_LIBS_DIR) $(HDF5_LIBS)

Indices_neighbors.o: Indices_neighbors.cc Parameters.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -c Indices_neighbors.cc

Main.o: Main.cc include/Check_parameters.hh include/Declare_variables.hh include/Declare_functions.hh include/Macros.hh Parameters.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -I$(HDF5_INC_DIR) -c Main.cc

Write_lattice.o: Write_lattice.cc include/Declare_variables.hh include/Declare_functions.hh include/Macros.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -I$(HDF5_INC_DIR) -c Write_lattice.cc

Update.o: Update.cc include/Declare_variables.hh include/Declare_functions.hh include/Macros.hh Parameters.hh
	$(CXX) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) $(CXX_WARN_FLAGS) $(CXX_DEBUG_FLAGS) -I$(HDF5_INC_DIR) -c Update.cc



# Remove the executable and all object files
# NOTE: icpx generates *.o.tmp files which should be removed when cleaning
.PHONY : clean
clean:
	${RM} ${EXE} ${OBJ} *.o.tmp
