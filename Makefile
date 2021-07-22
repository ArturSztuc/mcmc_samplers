UNAME := $(shell uname)

LIB_EXT="so"

all:
# Check that bin directory is made
ifeq "$(wildcard bin)" ""
	make links
endif

ifeq "$(wildcard lib)" ""
	make links
endif

	make -j23 -C mcmc
#	make -j23 -C Prob3++ shared
	make -j23 -C model
#	make -j23 -C utils
#	make -j23 -C diagnostics
	make -j23 -C src

links:
	@ [ -d lib ]     || mkdir lib
	@ [ -d bin ]     || mkdir bin

# The usual clean
clean:
	rm -rf lib/lib*
	rm -rf bin/*

# very clean
vclean:
	make vclean -C mcmc
	#make clean -C Prob3++
	make vclean -C model 
	#make vclean -C utils
	#make vclean -C diagnostics
	make vclean -C src
