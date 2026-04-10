CC=h5cc

all:
	$(CC) -Wall -O3 hdf5bench.c -lm -o hdf5bench
