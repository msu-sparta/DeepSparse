CC=g++
CPPFLAGS= -DdagP_METIS
CFLAGS = -O3 -w -fopenmp -m64



EXEDIR_DAG=../exe

lobpcg_global: 
	$(CC)  -c ../../common/util.cpp -fopenmp
	$(CC)  -c lobpcg_global.cpp -fopenmp
	$(CC) -o $(EXEDIR_DAG)/lobpcg_global.x util.o lobpcg_global.o -fopenmp



AR=ar rc
RANLIB=ranlib

#nersc
METISLIBDIR=/usr/common/software/metis/5.1.0/lib
METISHDIR=/usr/common/software/metis/5.1.0/include

INCDIR=../../common/partitioner/part_inc
SRCDIR=../../common/partitioner/part_src
LIBDIR=../../common/partitioner/part_lib


DGRAPHOBJS= $(SRCDIR)/dgraph.o $(SRCDIR)/dgraphTraversal.o $(SRCDIR)/utils.o $(SRCDIR)/info.o $(SRCDIR)/undirPartitioning.o $(SRCDIR)/option.o $(SRCDIR)/debug.o $(SRCDIR)/clustering.o $(SRCDIR)/dgraphReader.o $(SRCDIR)/dgraphDotReader.o $(SRCDIR)/vcycle2way.o $(SRCDIR)/initialBisection.o $(SRCDIR)/rvcycle.o $(SRCDIR)/dgraphBisection.o $(SRCDIR)/refinementBis.o


$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@ -I$(METISHDIR) -I$(INCDIR) 


all: $(EXEDIR_DAG)/lobpcg_part.x


$(LIBDIR)/libdgraph.a: $(DGRAPHOBJS)
	$(AR) $@ $(SRCDIR)/dgraph.o $(SRCDIR)/dgraphTraversal.o $(SRCDIR)/utils.o $(SRCDIR)/info.o $(SRCDIR)/undirPartitioning.o $(SRCDIR)/option.o $(SRCDIR)/debug.o $(SRCDIR)/clustering.o $(SRCDIR)/dgraphReader.o $(SRCDIR)/dgraphDotReader.o $(SRCDIR)/vcycle2way.o $(SRCDIR)/initialBisection.o $(SRCDIR)/rvcycle.o $(SRCDIR)/dgraphBisection.o $(SRCDIR)/refinementBis.o
	$(RANLIB) $@


$(EXEDIR_DAG)/lobpcg_part.x: $(SRCDIR)/rMLGP.o $(LIBDIR)/libdgraph.a
	$(CC) $(CPPFLAGS) $(CFLAGS) -c ../../common/util.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) -c lobpcg_part.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(EXEDIR_DAG)/lobpcg_part.x -static lobpcg_part.o $(SRCDIR)/rMLGP.o util.o -L$(METISLIBDIR) -L$(LIBDIR) -lmetis -ldgraph -lm


clean: 
	rm -f *.o ../../common/*.o ../../common/partitioner/part_src/*.o ../../common/partitioner/part_lib/*.a
