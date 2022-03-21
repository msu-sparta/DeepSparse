CC=mpic++
CFLAGS=-march=native -fopenmp -O3 -m64 -I${MKL_ROOT}/include
CFLAGS_EXTRA=-Wall -Wextra -Wconversion
LINKER_DYNAMIC=-L${MKL_ROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

EXEDIR=.

sparse: 
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -o $(EXEDIR)/custom_power_iteration.x -g $(LINKER_DYNAMIC) custom_power_iteration.cpp
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -o $(EXEDIR)/mkl_power_iteration.x -g $(LINKER_DYNAMIC) mkl_power_iteration.cpp

clean:
	rm $(EXEDIR)/custom_power_iteration.x $(EXEDIR)/mkl_power_iteration.x
