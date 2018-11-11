MAIN=pMain
CUDACC=nvcc
CPPFLAGS=-std=c++11
CUDACCFLAGS=-std c++11 --expt-extended-lambda -x cu --gpu-architecture compute_35 -lcuda -lcudart
OUTDIR=../bin/

#COMPILE
$(MAIN): $(MAIN).cu
	$(CUDACC) $(MAIN).cu $(CUDACCFLAGS) -o $(OUTDIR)$(MAIN)
