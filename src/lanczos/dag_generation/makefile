CC=icpc
CFLAGS=-std=c++11 -Wall -Wextra -Wconversion -O3 -xmic-avx512 -qopenmp

EXEDIR=../exe

lanczos_global: 
	$(CC)  -c ../../common/util.cpp  -qopenmp
	$(CC)  -c lanczos_global.cpp   util.o -qopenmp
	$(CC) -o $(EXEDIR)/lanczos_global.x util.o lanczos_global.o -qopenmp


clean: 
	rm -rf *.o 