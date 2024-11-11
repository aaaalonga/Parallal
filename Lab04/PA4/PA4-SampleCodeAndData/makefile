CC = /usr/bin/gcc
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -Xcompiler -g -w -I..  
MPI = -I /usr/local/software/openmpi/include/ 
LIB = -L /usr/local/software/openmpi/lib -lmpi  

BINS:= multiple

multiple: multiple.cu
	${NVCC} ${NVCCFLAGS} ${MPI} ${LIB} --std=c++11 multiple.cu -o multiple


test_small_1: $(BINS)
	@mpiexec -n 4 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./multiple 4 512 small_input_data_3_1.txt
test_small_2: $(BINS)
	@mpiexec -n 3 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./multiple 1 512 small_input_data_3_3.txt
test_small_3: $(BINS)
	@mpiexec -n 4 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./multiple 1 512 small_input_data_3_5.txt

test_large_1: $(BINS)
	@mpiexec -n 2 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./multiple 4 128 large_input_data_200_10.txt
test_large_2: $(BINS)
	@mpiexec -n 3 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./multiple 2 64 large_input_data_299_29.txt
test_large_3: $(BINS)
	@mpiexec -n 4 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./multiple 1 512 large_input_data_599_3.txt

clean:
	-rm multiple


