CC=g++
CFLAGS=-std=c++11 -Wall -Wextra -Wconversion -O3 -xmic-avx512 -qopenmp

EXEDIR=../exe

nonloop_global: 
	$(CC)  -c ../../common/util.cpp  -fopenmp
	$(CC)  -c lobpcg_global.cpp   util.o -fopenmp
	$(CC) -o $(EXEDIR)/lobpcg_global.x util.o lobpcg_global.o -fopenmp


clean: 
	rm -rf *.o 
