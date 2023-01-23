CXX=g++
AR=ar
LD_SHARED=g++

ROOTLIBS := $(shell root-config --glibs --libs)
ROOTINCLUDES := $(shell root-config --prefix)/include 
ROOTCFLAGS := $(shell root-config --cflags) -I../

CUDA_LIBS := -L/opt/cuda/lib64 -I/opt/cuda/include -lcudart
LIB_OBJECTS += $(ROOTLIBS)
LIB_OBJECTS += $(CUDA_LIBS)

INCDIR = -I /usr/include/eigen3
LIBDIR = ../lib

UNAME := $(shell uname)

CXXFLAGS   = -Wall -O3 -g -fPIC $(ROOTCFLAGS) $(INCDIR) -Werror -fopenmp
CUDA_CXXFLAGS=-prec-sqrt=false -use_fast_math -O3 -Werror cross-execution-space-call -allow-unsupported-compiler
CUDA_PTX="-dlcm=ca -allow-expensive-optimizations=true -fmad=true -O3 -warn-lmem-usage -warn-spills"
#CUDA_XCOMP="-fpic -O3 -Wall -Wextra -Werror" -c
CUDA_XCOMP="-fpic -O3 -Wall -Werror" -c

#CXXFLAGS   = -Wall -g -fPIC $(ROOTCFLAGS) $(INCDIR) -Werror -fopenmp
SYSLIB     = -lm -lg2c
LINK_ARGS_BIN = $(SYSLIB) $(LIBDIR)
SOFLAGS= -shared $(ROOTCFLAGS)
ARFLAGS= rcsv

OBJECTS := diagnostics.o diag_traces.o diag_autoc.o diag_posterior.o cuda_diagnostics.o #diag_skspectra.o diag_postfitcov.o diag_importancesamp.o diag_rhat.o

CUDA_OBJECTS := cuda_diag_kernel.o

%.o:%.cu
	nvcc -c $(CUDA_CXXFLAGS) \
		-Xptxas $(CUDA_PTX) \
		-Xcompiler $(CUDA_XCOMP)\
		$(CUDA_LIBS) \
		$<

%.o:%.cpp
	g++ -c $(CXXFLAGS) $< $(OMP_DEFINES)

libDiagnostics: libDiagnostics.a $(LIBDIR)/libDiagnostics.so

libDiagnostics.a: $(OBJECTS) $(CUDA_OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(LIBDIR)/libDiagnostics.so: libDiagnostics.a $(OBJECTS) $(CUDA_OBJECTS)
	$(LD_SHARED) $(SOFLAGS) $^ -o $@ $(LIB_OBJECTS) $(ROOTLIBS) -L$(LIBDIR)

clean:
	rm -f ../lib/libDiagnostics.so	
	rm -f *.o lib*

vclean:
	rm -f ../lib/libDiagnostics.so	
	rm -f *.o lib*
	rm -fR ../lib

