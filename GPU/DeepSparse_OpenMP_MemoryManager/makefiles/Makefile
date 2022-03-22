CC=CC -DUSE_CUBLAS -DUSE_DEVICE -DUSE_LAPACK -DTIMER 
CPPFLAGS=-h omp -O3 -h std=c++11 -h noacc
EXEDIR=../bin

CUFLAGS=
CFLAGS=-I${CUDA_ROOT}/include
LIBS=-L${CUDA_ROOT}/lib64 -lcublas -lcudart -lcusparse -lnvToolsExt
SRCDIR=../src

LINKER_DYNAMIC=
# change this path with your own compiled LAPACK and CBLAS path
INC= -I/global/homes/r/rabbimd/lapack-3.8.0/LAPACKE/include -I/global/homes/r/rabbimd/CBLAS/include
LIB=-L/global/homes/r/rabbimd/lapack-3.8.0/lib -L/global/homes/r/rabbimd/CBLAS/lib -llapacke -llapack -lcblas -lblas -lgfortran -lm

all:
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LIBS) -c  $(SRCDIR)/util.cpp -o $(EXEDIR)/util.o $(LINKER_DYNAMIC) $(INC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LIBS) -c  $(SRCDIR)/matrix_ops_cpu.cpp -o $(EXEDIR)/matrix_ops_cpu.o $(LINKER_DYNAMIC) $(INC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LIBS) -c  $(SRCDIR)/matrix_ops_gpu.cpp -o $(EXEDIR)/matrix_ops_gpu.o $(LINKER_DYNAMIC) $(INC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LIBS) -c  $(SRCDIR)/memory_manager.cpp -o $(EXEDIR)/memory_manager.o $(LINKER_DYNAMIC) $(INC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LIBS) -c  $(SRCDIR)/lobpcg_gpu_mm.cpp -o $(EXEDIR)/lobpcg_gpu_mm.o $(LINKER_DYNAMIC) $(INC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LIBS)  $(EXEDIR)/util.o $(EXEDIR)/matrix_ops_cpu.o $(EXEDIR)/matrix_ops_gpu.o $(EXEDIR)/memory_manager.o $(EXEDIR)/lobpcg_gpu_mm.o -o $(EXEDIR)/lobpcg_gpu_mm.x $(LINKER_DYNAMIC) $(INC) $(LIB)

clean:
	rm $(EXEDIR)/*.x $(EXEDIR)/*.o
