//
// uenv start  prgenv-gnu/25.11:v1 --view=default
// h5cc -o hdf5bench hdf5bench.c
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <hdf5.h>

const float GiB = 1024*1024*1024;

#define FILENAME "benchmark"
#define DATASET_NAME "field"

void usage(int rank) {
    if (rank == 0) {
        printf("Usage:\n");
        printf("  -m [shared|perrank]\n");
        printf("  -t total_size_bytes OR -p per_rank_size_bytes\n");
        printf("  -f num_fields\n");
        printf("  [-a alignment_bytes]\n");
        printf("  [-T alignment_threshold_bytes]\n");
        printf("  [-c chunk_elements]\n");
        printf("  [-o output_prefix]\n");
        printf("  [-i num_iterations]\n");
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size, rank_comm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char mode[16] = "shared";
    size_t total_size = 0, per_rank_size = 0;
    int num_fields = 1;

    size_t alignment = 0, threshold = 0;
    hsize_t chunk_size = 0;

    char output_prefix[4096-256] = "benchmark";
    char filename[4096];

    int iterations = 1;

    // ---------------------------
    // Parse args
    // ---------------------------
    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "-m") == 0) strncpy(mode, argv[++i], sizeof(mode)-1);
        else if (strcmp(argv[i], "-t") == 0) total_size = atoll(argv[++i]);
        else if (strcmp(argv[i], "-p") == 0) per_rank_size = atoll(argv[++i]);
        else if (strcmp(argv[i], "-f") == 0) num_fields = atoi(argv[++i]);
        else if (strcmp(argv[i], "-a") == 0) alignment = atoll(argv[++i]);
        else if (strcmp(argv[i], "-T") == 0) threshold = atoll(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0) chunk_size = atoll(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-o") == 0) strncpy(output_prefix, argv[++i], sizeof(output_prefix)-1);
    }

    if ((total_size == 0 && per_rank_size == 0) ||
        (total_size && per_rank_size)) {
        usage(rank);
        MPI_Finalize();
        return 1;
    }

    if (total_size > 0)
        per_rank_size = total_size / size;

    size_t elems_rank = per_rank_size / sizeof(double);
    size_t elems_field = elems_rank / num_fields;

    double *data = malloc(elems_field * sizeof(double));
    for (size_t i = 0; i < elems_field; i++)
        data[i] = rank + i * 1e-3;


    if (strcmp(mode, "perrank") == 0)
        MPI_Comm_split(MPI_COMM_WORLD, rank, 0, &rank_comm);
    else
        rank_comm = MPI_COMM_WORLD;

    if (strcmp(mode, "perrank") == 0)
        sprintf(filename, "%s_rank_%d.h5", output_prefix, rank);
    else
        sprintf(filename, "%s_shared.h5", output_prefix);

    for (int iter = 1; iter <= iterations; iter++) {

        // ---------------------------
        // HDF5 setup
        // ---------------------------
        hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist, rank_comm, MPI_INFO_NULL);

        H5Pset_file_locking(plist, 0, 0);

        if (alignment > 0)
            H5Pset_alignment(plist, threshold, alignment);


        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

        hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, plist);
        MPI_Barrier(MPI_COMM_WORLD);

        H5Pclose(plist);

        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        if (strcmp(mode, "perrank") == 0)
            H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT);
        else
            H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

        double t_start, t_create = 0, t_write = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        t_start = MPI_Wtime();

        for (int f = 0; f < num_fields; f++) {
            char name[64];
            sprintf(name, "%s_%d", DATASET_NAME, f);

            hsize_t gsize, lsize, start;

            if (strcmp(mode, "shared") == 0) {
                gsize = elems_field * size;
                lsize = elems_field;
                start = rank * elems_field;
            } else {
                gsize = elems_field;
                lsize = elems_field;
                start = 0;
            }

            hid_t filespace = H5Screate_simple(1, &gsize, NULL);

            // dataset creation properties
            hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
            if (chunk_size > 0)
                H5Pset_chunk(dcpl, 1, &chunk_size);

            double t1 = MPI_Wtime();
            hid_t dset = H5Dcreate(file, name, H5T_NATIVE_DOUBLE,
                                   filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            t_create += MPI_Wtime() - t1;

            H5Pclose(dcpl);

            H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                &start, NULL, &lsize, NULL);

            hid_t memspace = H5Screate_simple(1, &lsize, NULL);

            t1 = MPI_Wtime();
            H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, data);
            t_write += MPI_Wtime() - t1;

            H5Sclose(memspace);
            H5Sclose(filespace);
            H5Dclose(dset);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t_write_end = MPI_Wtime();

        H5Fflush(file, H5F_SCOPE_GLOBAL);

        MPI_Barrier(MPI_COMM_WORLD);
        double t_flush_end = MPI_Wtime();

        H5Pclose(dxpl);
        H5Fclose(file);

        MPI_Barrier(MPI_COMM_WORLD);
        double t_close_end = MPI_Wtime();
        double t_end = MPI_Wtime();

        // ---------------------------
        // REDUCTION
        // ---------------------------
        double max_create, max_write, max_flush, max_close, max_total;

        double flush_time = t_flush_end - t_write_end;
        double close_time = t_close_end - t_flush_end;
        double total_time = t_end - t_start;

        MPI_Reduce(&t_create,   &max_create, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&t_write,    &max_write,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&flush_time, &max_flush,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&close_time, &max_close,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_time, &max_total,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double total_bytes = (double)per_rank_size * size;

        if (rank == 0) {
            printf("===== BEGIN REPORT =====\n");

            printf("Mode: %s\nRanks: %d\nFields: %d\n", mode, size, num_fields);
            printf("Per-rank size: %zu bytes\n", per_rank_size);
            if (alignment) printf("Alignment: %zu (threshold %zu)\n", alignment, threshold);
            if (chunk_size) printf("Chunk size: %llu elements\n", (unsigned long long)chunk_size);

            printf("Iteration: %i of %i\n", iter, iterations);
            printf("Total data: %.3f GiB\n", total_bytes / GiB);
            printf("Output prefix: %s\n", output_prefix);

            printf("\nCreate time: %.6f s\n", max_create);
            printf("Write time:  %.6f s\n", max_write);
            printf("Flush time:  %.6f s\n", max_flush);
            printf("Close time:  %.6f s\n", max_close);
            printf("Total time:  %.6f s\n", max_total);

            printf("\nBandwidth (write only):        %.3f GiB/s\n", total_bytes / max_write / GiB);
            printf("Bandwidth (write+flush):       %.3f GiB/s\n",
                   total_bytes / (max_write + max_flush) / GiB);
            printf("Bandwidth (write+flush+close): %.3f GiB/s\n",
                   total_bytes / (max_write + max_flush + max_close) / GiB);
            printf("Bandwidth (total):             %.3f GiB/s\n",
                   total_bytes / max_total / GiB);
            printf("===== END REPORT =====\n\n");
        }
    }

    free(data);

    MPI_Finalize();
    return 0;
}

