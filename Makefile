# Include the option list, but don't do that when running 'make clean'
ifeq (, $(filter clean, $(MAKECMDGOALS)))
	include $(options)
endif


# Objects to be built
OBJ = Main.o Check_parameters.o Update.o Thermalize.o
EXE = Ising2D


# Disable optimization if so requested by the user
ifneq ($(OPTIMIZE), yes)
	CXX_OPTIMIZE_FLAGS =
endif


# icpx generates *.o.tmp files which should be removed when cleaning
ifeq ($(CXX), icpx)
	ICPX_OTMP = *.o.tmp
else
	ICPX_OTMP =
endif


# Build all targets
build: $(OBJ)
	$(CXX) -o $(EXE) $(OBJ) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) -I$(HDF5_INC_DIR) -L$(HDF5_LIBS_DIR) $(HDF5_LIBS)

Main.o: Main.cc
	$(CXX) -c Main.cc $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) -I$(HDF5_INC_DIR)

Check_parameters.o: Check_parameters.cc
	$(CXX) -c Check_parameters.cc $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) -I$(HDF5_INC_DIR)

Update.o: Update.cc
	$(CXX) -c Update.cc $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) -I$(HDF5_INC_DIR)

Thermalize.o: Thermalize.cc
	$(CXX) -c Thermalize.cc $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS) -I$(HDF5_INC_DIR)

.PHONY : clean
clean:
	$(RM) $(EXE) $(OBJ) $(ICPX_OTMP)
