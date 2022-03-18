#CC=icpc
CC=g++
#CFLAGS=-std=c++11 -Wall -Wextra -Wconversion -O3 -xmic-avx512 -qopenmp
CFLAGS=-std=c++11 -Wall -Wextra -Wconversion -O3 -march=native -fopenmp

EXEDIR=../exe

lanczos_global: 
	#$(CC)  -c ../../common/util.cpp  -qopenmp
	#$(CC)  -c lanczos_global.cpp   util.o -qopenmp
	#$(CC) -o $(EXEDIR)/lanczos_global.x util.o lanczos_global.o -qopenmp
	$(CC)  -c ../../common/util.cpp  -fopenmp
	$(CC)  -c lanczos_global.cpp -fopenmp
	$(CC) -o $(EXEDIR)/lanczos_global.x util.o lanczos_global.o -fopenmp

clean: 
	rm -rf *.o 
