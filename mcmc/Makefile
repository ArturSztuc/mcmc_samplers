CXX=g++
AR=ar
LD_SHARED=g++

ROOTLIBS := $(shell root-config --glibs --libs)
ROOTINCLUDES := $(shell root-config --prefix)/include 
ROOTCFLAGS := $(shell root-config --cflags) -I../

INCDIR = -I /usr/include/eigen3
LIBDIR = ../lib

UNAME := $(shell uname)

CXXFLAGS   = -Wall -O3 -g -fPIC $(ROOTCFLAGS) $(INCDIR) -Werror
SYSLIB     = -lm -lg2c
LINK_ARGS_BIN = $(SYSLIB) $(LIBDIR)
SOFLAGS= -shared $(ROOTCFLAGS)
ARFLAGS= rcsv

OBJECTS := mcmc_methastings.o # mcmc_hamiltonian.o mcmc_temphamiltonian.o mcmc_paralleltempering.o mcmc_hammer.o mcmc_manager.o mcmc_reversejump.o

%.o:%.cpp
	g++ -c $(CXXFLAGS) $< $(OMP_DEFINES)

libMCMC: libMCMC.a $(LIBDIR)/libMCMC.so

libMCMC.a: $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(LIBDIR)/libMCMC.so: libMCMC.a $(OBJECTS)
	$(LD_SHARED) $(SOFLAGS) $^ -o $@ $(LIB_OBJECTS) $(ROOTLIBS) -L$(LIBDIR)

clean:
	rm -f ../lib/libMCMC.so	
	rm -f *.o lib*

vclean:
	rm -f ../lib/libMCMC.so	
	rm -f *.o lib*
	rm -fR ../lib

