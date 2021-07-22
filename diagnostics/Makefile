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

OBJECTS := diag_autoc.o diag_traces.o diag_skspectra.o diag_postfitcov.o diag_importancesamp.o diag_rhat.o

%.o:%.cpp
	g++ -c $(CXXFLAGS) $< $(OMP_DEFINES)

libDiagnostics: libDiagnostics.a $(LIBDIR)/libDiagnostics.so

libDiagnostics.a: $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(LIBDIR)/libDiagnostics.so: libDiagnostics.a $(OBJECTS)
	$(LD_SHARED) $(SOFLAGS) $^ -o $@ $(LIB_OBJECTS) $(ROOTLIBS) -L$(LIBDIR) -lThreeProb_1.00

clean:
	rm -f ../lib/libDiagnostics.so	
	rm -f *.o lib*

vclean:
	rm -f ../lib/libDiagnostics.so	
	rm -f *.o lib*
	rm -fR ../lib

