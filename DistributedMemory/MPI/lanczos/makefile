CC=mpic++
CFLAGS=-march=native -fopenmp -O3 -m64 -I${MKL_ROOT}/include
CFLAGS_EXTRA=-Wall -Wextra -Wconversion
LINKER_DYNAMIC=-L${MKL_ROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

EXEDIR=.

sparse: 
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -o $(EXEDIR)/custom_lanczos.x -g $(LINKER_DYNAMIC) custom_lanczos.cpp
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -o $(EXEDIR)/mkl_lanczos.x -g $(LINKER_DYNAMIC) mkl_lanczos.cpp

clean:
	rm $(EXEDIR)/custom_lanczos.x $(EXEDIR)/mkl_lanczos.x
