CC=icpc
CFLAGS=-std=c++11 -Wall -Wextra -Wconversion -O3 -xmic-avx512 -qopenmp

EXEDIR=../exe

nonloop_global: 
	$(CC)  -c ../../common/util.cpp  -qopenmp
	$(CC)  -c lobpcg_global.cpp   util.o -qopenmp
	$(CC) -o $(EXEDIR)/lobpcg_global.x util.o lobpcg_global.o -qopenmp


clean: 
	rm -rf *.o 