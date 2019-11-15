CC=icpc
CFLAGS=-std=c++11 -Wall -Wextra -Wconversion -O3 -xmic-avx512 -qopenmp

EXEDIR=../exe

lanczos_global: 
	$(CC)  -c ../../common/util.cpp  -qopenmp
	$(CC)  -c power_global.cpp   util.o -qopenmp
	$(CC) -o $(EXEDIR)/power_global.x util.o power_global.o -qopenmp


clean: 
	rm -rf *.o 