PMAIN=pMain
SERVER=serverMain
CUDACC=nvcc
CPP=g++
CPPFLAGS=-std=c++11
CUDACCFLAGS=-std c++11 --expt-extended-lambda -x cu --gpu-architecture compute_35 -lcuda -lcudart
OUTDIR=../bin/

#all: $(PMAIN) 
all: $(PMAIN) $(SERVER)
	
#COMPILE
# Paralel and Sequential Network Tests
$(PMAIN): $(PMAIN).cu
	$(CUDACC) $(PMAIN).cu $(CUDACCFLAGS) -o $(OUTDIR)$(PMAIN)

#$(CUDACC) $(SERVER).cu $(CUDACCFLAGS) -o $(OUTDIR)$(SERVER) 
# Server 
$(SERVER): $(SERVER).cu
	$(CUDACC) $(SERVER).cu $(CUDACCFLAGS) -o $(OUTDIR)$(SERVER) 
