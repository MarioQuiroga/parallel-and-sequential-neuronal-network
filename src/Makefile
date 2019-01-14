PMAIN=pMain
SMAIN=sMain
SERVER=serverMain
CUDACC=nvcc
CPP=g++
CPPFLAGS=-std=c++11
CUDACCFLAGS=-std c++11 --expt-extended-lambda -x cu --gpu-architecture compute_35 -lcuda -lcudart
OUTDIR=../bin/

#all: $(SMAIN)S
all: $(PMAIN) $(SMAIN)
#all: $(PMAIN) $(SMAIN) $(SERVER)

#COMPILE
# Paralel Network
$(PMAIN): $(PMAIN).cu
	$(CUDACC) $(PMAIN).cu $(CUDACCFLAGS) -o $(OUTDIR)$(PMAIN)

# Sequential Network
$(SMAIN): $(SMAIN).cpp
	$(CPP) $(SMAIN).cpp $(CPPFLAGS) -o $(OUTDIR)$(SMAIN)

# Server 
$(SERVER): $(SERVER).cu
	$(CUDACC) $(SERVER).cu $(CUDACCFLAGS) -o $(OUTDIR)$(SERVER) 
