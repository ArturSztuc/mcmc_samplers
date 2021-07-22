SRCS = $(wildcard *.cpp)
TRGS = $(patsubst %.cpp, ../bin/%, $(SRCS))

all : $(TRGS)

# General CPPFLAGS
OPTS=-O3 -Wno-write-strings -g

ROOTLIBS=$(shell root-config --libs)
ROOTCFLAGS=$(shell root-config --cflags) 

LIBDIR = ../lib/
INCLUDES=-I../ -I/usr/include/eigen3

# Project libraries
MYLIBS=-lMCMC -lModel #-lPriors -lDiagnostics -lUtils -lThreeProb_1.00 

# All the CPPFLAGS combines
CPPFLAGS=$(OPTS) $(ROOTCFLAGS)

# All the libraries combined
LIBS=$(MYLIBS) $(ROOTLIBS)

# Are we using CUDA? Nope, but we can dream.
ifdef CUDAPATH
CUDA_DEFINES=-DCUDA
endif

# Multithreading? Only available in some model and ensemble-mcmc classes
ifdef MULTITHREAD
OMP_DEFINES=-DMULTITHREAD
CPPFLAGS+=-fopenmp
endif

$(TRGS) : $(SRCS)

# Rather ugly. At the moment you either have to set LD_LIBRARY_PATH (which I
# hate), or run from one of the project's folders. Or insert the compiled
# libraries into one of the system library folders?
../bin/% : %.cpp
	g++ $(CPPFLAGS) $(INCLUDES) -o $@ $< -L$(LIBDIR) $(LIBS) -Wl,-R$(LIBDIR) '-Wl,-R./lib' $(CUDA_DEFINES) $(OMP_DEFINES)

# Remove the executables
clean:
	rm -f ../bin/*

# Very clean
vclean:
	rm rm -f ../bin/*
	rm -fR ../bin
