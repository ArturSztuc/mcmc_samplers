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

OBJECTS := model_covariance.o #model_covid.o model_oscprob.o model_multimodel.o model_simpleneutrinoexp.o model_bestpizza.o model_skdetassymetry.o model_mmonbeam.o model_raytracereco.o

%.o:%.cpp
	g++ -c $(CXXFLAGS) $< $(OMP_DEFINES)

libModel: libModel.a $(LIBDIR)/libModel.so

libModel.a: $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(LIBDIR)/libModel.so: libModel.a $(OBJECTS)
	$(LD_SHARED) $(SOFLAGS) $^ -o $@ $(LIB_OBJECTS) $(ROOTLIBS) -L$(LIBDIR) #-lThreeProb_1.00

clean:
	rm -f ../lib/libModel.so	
	rm -f *.o lib*

vclean:
	rm -f ../lib/libModel.so	
	rm -f *.o lib* *.a
	rm -f ../lib
