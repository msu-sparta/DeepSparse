#default compilers for each system

CC_NERSC=icpc
CC_HPCC=g++
CPPFLAGS_NERSC=-std=c++11 -Wall -Wextra -Wconversion -O3 -xmic-avx512 -qopenmp
CPPFLAGS_HPCC=-O3 -w -fopenmp -m64 -fopenmp


EXEDIR=../exe

lanczos_global_hpcc: 
	$(CC_HPCC) $(CPPFLAGS_HPCC) ../../common/util.cpp -o util.o -c
	$(CC_HPCC) $(CPPFLAGS_HPCC) power_global.cpp -o power_global.o -c
	$(CC_HPCC) $(CPPFLAGS_HPCC) -o $(EXEDIR)/power_global.x util.o power_global.o

lanczos_global_nersc:
	$(CC_NERSC) $(CPPFLAGS_NERSC) -c ../../common/util.cpp
	$(CC_NERSC) $(CPPFLAGS_NERSC) -c power_global.cpp   util.o
	$(CC_NERSC) $(CPPFLAGS_NERSC) -o $(EXEDIR)/power_global.x util.o power_global.o

clean: 
	rm -rf *.o $(EXEDIR)/power_global.x
