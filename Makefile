CC=h5cc

all:
	$(CC) -Wall -O3 hdf5bench.c -o hdf5bench
