# Include the option list, but don't do that when running 'make clean'
ifeq (, $(filter clean, $(MAKECMDGOALS)))
	include $(options)
endif


# Objects to be built
OBJ = Main.o Check_parameters.o
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
	$(CXX) -o $(EXE) $(OBJ) $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS)

Main.o: Main.cc
	$(CXX) -c Main.cc $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS)

Check_parameters.o: Check_parameters.cc
	$(CXX) -c Check_parameters.cc $(CXXFLAGS) $(CXX_OPTIMIZE_FLAGS)

.PHONY : clean
clean:
	$(RM) $(EXE) $(OBJ) $(ICPX_OTMP)
