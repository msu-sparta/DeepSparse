CC=g++
CFLAGS=-march=native -fopenmp -O3 -m64 -I${MKL_ROOT}/include
CFLAGS+=-DSHORT_INT
CFLAGS_EXTRA=-Wall -Wextra -Wconversion
LINKER_DYNAMIC=-L${MKL_ROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

EXEDIR=../exe

power_libcsr: 
	$(CC) $(CFLAGS) -c ../../common/exec_util.cpp -g $(LINKER_DYNMAIC)
	$(CC) $(CFLAGS) -c ../../common/matrix_ops.cpp -g $(LINKER_DYNAMIC)
	$(CC) $(CFLAGS) -c ../../common/vector_ops.cpp -g $(LINKER_DYNAMIC)
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -c power_libcsr_executor.cpp -g $(LINKER_DYNAMIC)
	$(CC) $(CFLAGS) -o $(EXEDIR)/power_libcsr_executor.x exec_util.o power_libcsr_executor.o matrix_ops.o vector_ops.o  -g $(LINKER_DYNAMIC)

power_libcsb: 
	$(CC) $(CFLAGS) -c ../../common/exec_util.cpp -g $(LINKER_DYNMAIC)
	$(CC) $(CFLAGS) -c ../../common/matrix_ops.cpp -g $(LINKER_DYNAMIC)
	$(CC) $(CFLAGS) -c ../../common/vector_ops.cpp -g $(LINKER_DYNAMIC)
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -c power_libcsb_executor.cpp -g $(LINKER_DYNAMIC)
	$(CC) $(CFLAGS) -o $(EXEDIR)/power_libcsb_executor.x exec_util.o power_libcsb_executor.o matrix_ops.o vector_ops.o -g $(LINKER_DYNAMIC)

power_global: 
	$(CC) $(CFLAGS) -c ../../common/exec_util.cpp -g $(LINKER_DYNMAIC)
	$(CC) $(CFLAGS) -c ../../common/matrix_ops.cpp -g $(LINKER_DYNAMIC)
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -c ../../common/vector_ops.cpp -g $(LINKER_DYNAMIC)
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -c power_global_executor.cpp -g $(LINKER_DYNAMIC)
	$(CC) $(CFLAGS) -o $(EXEDIR)/power_global_executor.x exec_util.o power_global_executor.o matrix_ops.o vector_ops.o -g $(LINKER_DYNAMIC)

clean:
	rm -rf *.p $(EXEDIR)/power_*_executor.x
