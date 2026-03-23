# hdf5bench

A minimal MPI + HDF5 benchmark for measuring parallel I/O performance on HPC systems.

## Overview

`hdf5bench` is a lightweight benchmarking tool designed to evaluate the
performance of parallel HDF5 writes under controlled conditions. It
focuses on a simple and transparent access pattern:

-   Multiple MPI ranks
-   Writing to a shared HDF5 file
-   Fixed per-rank data size
-   Basic dataset layout

The goal is to provide clear, reproducible measurements of:

-   Dataset creation time
-   Write bandwidth
-   Flush/synchronization cost
-   End-to-end I/O time

## Features

-   MPI-parallel HDF5 I/O
-   Shared-file access pattern
-   Configurable:
    -   Number of ranks
    -   Number of datasets (fields)
    -   Per-rank data size
    -   Number of iterations
-   Simple output format for scripting and analysis
-   No external dependencies beyond MPI and HDF5

## Build

### Requirements

-   MPI (e.g., OpenMPI, MPICH, Cray MPI)
-   Parallel HDF5 (built with MPI support)

### Compile

``` bash
h5cc -o hdf5bench hdf5bench.c
```

## Usage

``` bash
srun -n <ranks> ./hdf5bench [options]

where options are

  -m [shared|perrank]
  -t total_size_bytes OR -p per_rank_size_bytes
  -f num_fields
  [-a alignment_bytes]
  [-T alignment_threshold_bytes]
  [-c chunk_elements]
  [-o output_prefix]
  [-i num_iterations]
```

## Output

```
===== BEGIN REPORT =====
Mode: shared
Ranks: 512
Fields: 8
Per-rank size: 4294967296 bytes
Iteration: 1 of 10
Total data: 2048.000 GiB
Output prefix: bench.128

Create time: 0.000993 s
Write time:  89.528577 s
Flush time:  1.344479 s
Close time:  0.003056 s
Total time:  90.876799 s

Bandwidth (write only):        22.875 GiB/s
Bandwidth (write+flush):       22.537 GiB/s
Bandwidth (write+flush+close): 22.536 GiB/s
Bandwidth (total):             22.536 GiB/s
===== END REPORT =====
```

## I/O Pattern

The benchmark implements a shared-file parallel write:

-   One HDF5 file
-   Multiple datasets (fields)
-   Each rank writes its own contiguous hyperslab
-   Collective or independent I/O depending on implementation

## Use Cases

-   Comparing parallel filesystems (e.g., Lustre vs NFS)
-   Evaluating MPI-IO behavior under HDF5
-   Identifying metadata bottlenecks and flush latency

## Limitations

-   Single access pattern
-   No async I/O
-   No compression or advanced HDF5 features

