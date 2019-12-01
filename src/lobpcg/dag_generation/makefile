CC=icpc 
CPPFLAGS= -O3 -w -qopenmp #-DdagP_METIS
CFLAGS = -O3 -w -qopenmp #-DdagP_METIS



EXEDIR_DAG=../exe

#lobpcg_global: 
#	$(CC)  -c ../../common/util.cpp  -fopenmp
#	$(CC)  -c lobpcg_global.cpp   util.o -fopenmp
#	$(CC) -o $(EXEDIR_DAG)/lobpcg_global.x util.o lobpcg_global.o -fopenmp



LIBS=-ldgraph -lm
LIBSA=-lmetis $(LIBS)
AR=ar rv
RANLIB=ranlib

#nersc
#METISLIBDIR=/usr/common/software/metis/5.1.0/lib
#METISHDIR=/usr/common/software/metis/5.1.0/include

#hpcc
METISLIBDIR=/opt/software/METIS/5.1.0-intel-2017.01/lib
METISHDIR=/opt/software/METIS/5.1.0-intel-2017.01/include


INCDIR=../../common/partitioner/part_inc
SRCDIR=../../common/partitioner/part_src
EXEDIR=../../common/partitioner/part_exe
LIBDIR=../../common/partitioner/part_lib

LIBDIRA=-L$(LIBDIR) -L$(METISLIBDIR)
INCDIRA=-I$(INCDIR) -I$(METISHDIR)


BINS=$(EXEDIR)/metis $(EXEDIR)/rMLGP 
DGRAPHOBJS= $(SRCDIR)/dgraph.o $(SRCDIR)/dgraphTraversal.o $(SRCDIR)/utils.o $(SRCDIR)/info.o $(SRCDIR)/undirPartitioning.o $(SRCDIR)/option.o $(SRCDIR)/debug.o $(SRCDIR)/clustering.o $(SRCDIR)/dgraphReader.o $(SRCDIR)/dgraphDotReader.o $(SRCDIR)/vcycle2way.o $(SRCDIR)/initialBisection.o $(SRCDIR)/rvcycle.o $(SRCDIR)/dgraphBisection.o $(SRCDIR)/refinementBis.o


DGRAPHSRCS= $(DGRAPHOBJS:.o=.c)

DGRAPHLIB=$(LIBDIR)/libdgraph.a
#METISLIB=$(LIBDIR)/libmetis.a


#@echo "hello world"

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp $(INCDIR)/dgraph.h
	$(CC) -c $< -o $@ $(CFLAGS) $(INCDIRA) -I$(METISHDIR)


#$(SRCDIR)/%.o: $(SRCDIR)/%.cpp $(INCDIR)/dgraph.h
#	$(CXX) -c $< -o $@ $(CPPFLAGS) -std=c++11 -I$(INCDIR)

all: $(LIBDIR)/libdgraph.a $(EXEDIR)/rMLGP #$(EXEDIR)/metis


$(LIBDIR)/libdgraph.a: $(DGRAPHOBJS) $(INCDIR)/dgraph.h
	@mkdir -p $(@D)
	$(AR) $@ $(DGRAPHOBJS)
	$(RANLIB) $@

#$(EXEDIR)/metis: $(SRCDIR)/metis.o $(DGRAPHLIB)
#	@mkdir -p $(@D)
#	$(CC) -o $@ $^ $(CFLAGS) $(LIBDIRA) $(LIBSA) $(INCDIRA)

$(EXEDIR)/rMLGP: $(SRCDIR)/rMLGP.o $(DGRAPHLIB) 
	$(CC) $(CFLAGS) -c ../../common/util.cpp  
	$(CC) $(CFLAGS)  -c lobpcg_part.cpp util.o 
	$(CC) $(CFLAGS)  -o lobpcg_part.x lobpcg_part.o $(SRCDIR)/rMLGP.o $(LIBDIR)/libdgraph.a util.o  -L$(LIBDIR) -L$(METISLIBDIR)  -ldgraph -lm -I$(INCDIR) -I$(METISHDIR)
	#@mkdir -p $(@D)
	#$(CC) -o $@ $^ $(CFLAGS) $(LIBDIRA)  $(LIBSA) $(INCDIRA)






clean: 
	rm -rf *.o 
