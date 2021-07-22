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

OBJECTS := utils_contours1D.o utils_contours2D.o utils_triangle.o utils_beautify.o utils_pulls1D.o

%.o:%.cpp
	g++ -c $(CXXFLAGS) $< $(OMP_DEFINES)

libUtils: libUtils.a $(LIBDIR)/libUtils.so

libUtils.a: $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(LIBDIR)/libUtils.so: libUtils.a $(OBJECTS)
	$(LD_SHARED) $(SOFLAGS) $^ -o $@ $(LIB_OBJECTS) $(ROOTLIBS) -L$(LIBDIR) -lThreeProb_1.00

clean:
	rm -f ../lib/libUtils.so	
	rm -f *.o lib*

vclean:
	rm -f ../lib/libUtils.so	
	rm -f *.o lib*
	rm -fR ../lib

