//
// uenv start  prgenv-gnu/25.11:v1 --view=default
// h5cc -o hdf5bench hdf5bench.c
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <hdf5.h>
#include <math.h>

extern char **environ;

const float GiB = 1024*1024*1024;
const float GB  = 1000*1000*1000;

#define VERSION "1.1"
#define FILENAME "benchmark"
#define DATASET_NAME "field"

struct stats
{
    double min, max;
    double mean, std;
};

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
        printf("  [--printenv]\n");
    }
}

void collect_stats(double v, struct stats *stats, int N)
{
    double sum, sq_sum;
    double v2 = v * v;

    MPI_Reduce(&v,  &sum,       1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&v2, &sq_sum,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&v,  &stats->min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&v,  &stats->max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    stats->mean = sum / N;
    stats->std  = sqrt((sq_sum / N) - stats->mean * stats->mean);
}

int main(int argc, char **argv) {

    char mode[16] = "shared";
    char scaling[16] = "weak";
    size_t total_size = 0, per_rank_size = 0;
    int num_fields = 1;
    int print_env = 0;

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
        else if (strcmp(argv[i], "--printenv") == 0) print_env = 1;
    }

    if (print_env)
    {
        setenv("MPICH_MPIIO_HINTS_DISPLAY", "1", 1);
        setenv("MPICH_MPIIO_STATS", "1", 1);
    }
    MPI_Init(&argc, &argv);

    int rank, size, rank_comm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((total_size == 0 && per_rank_size == 0) ||
        (total_size && per_rank_size)) {
        usage(rank);
        MPI_Finalize();
        return 1;
    }

    if (print_env && rank == 0)
    {
        int found=0;
        for (char **env = environ; *env != NULL; env++) 
        {
            if (strstr(*env, "MPIIO") != NULL) 
            {
                printf("%s\n", *env);
                found=1;
            }
        }
        if (!found)
        {
            printf("NO MPIIO ENVVARS FOUND\n");
        }
    }


    if (total_size > 0)
    {
        per_rank_size = total_size / size;
        strcpy(scaling, "strong");
    }

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
        double t_bar_write = MPI_Wtime();

        double t_flush_start = MPI_Wtime();
        H5Fflush(file, H5F_SCOPE_GLOBAL);
        double t_flush = MPI_Wtime() - t_flush_start;

        MPI_Barrier(MPI_COMM_WORLD);
        double t_bar_flush = MPI_Wtime();

        double t_close_start = MPI_Wtime();
        H5Pclose(dxpl);
        H5Fclose(file);
        double t_close = MPI_Wtime() - t_close_start;

        MPI_Barrier(MPI_COMM_WORLD);
        double t_end = MPI_Wtime();

        // ---------------------------
        // REDUCTION
        // ---------------------------
        double t_total = t_end - t_start;
        struct stats stats_create, stats_write, stats_flush, stats_close;

        collect_stats(t_create, &stats_create, size);
        collect_stats(t_write,  &stats_write, size);
        collect_stats(t_flush,  &stats_flush, size);
        collect_stats(t_close,  &stats_close, size);

        double total_bytes = (double)per_rank_size * size;

        if (rank == 0) {
            printf("===== BEGIN REPORT v%s =====\n", VERSION);

            printf("Mode: %s\nScaling: %s\nRanks: %d\nFields: %d\n", mode, scaling, size, num_fields);
            printf("Output prefix: %s\n", output_prefix);
            printf("Per-rank size: %zu bytes = %.3f GB = %.3f GiB\n", per_rank_size, per_rank_size / GB, per_rank_size / GiB);
            if (alignment) printf("Alignment: %zu (threshold %zu)\n", alignment, threshold);
            if (chunk_size) printf("Chunk size: %llu elements\n", (unsigned long long)chunk_size);

            printf("Total data: %.3f GiB\n", total_bytes / GiB);
            printf("Iteration: %i of %i\n", iter, iterations);

            printf("\n");
            printf("Stats (max, min, mean, std)\n");
            printf("--------------------------------------------\n");
            printf("Create time: %11.6f s %11.6f %11.6f %11.6f\n", stats_create.max, stats_create.min, stats_create.mean, stats_create.std);
            printf(" Write time: %11.6f s %11.6f %11.6f %11.6f\n", stats_write.max,  stats_write.min,  stats_write.mean,  stats_write.std);
            printf(" Flush time: %11.6f s %11.6f %11.6f %11.6f\n", stats_flush.max,  stats_flush.min,  stats_flush.mean,  stats_flush.std);
            printf(" Close time: %11.6f s %11.6f %11.6f %11.6f\n", stats_close.max,  stats_close.min,  stats_close.mean,  stats_close.std);
            printf(" Total time: %11.6f s\n", t_total);

            printf("\n");
            printf("Bandwidth (create+write): %8.3f GiB/s\n", total_bytes / (t_bar_write - t_start) / GiB);
            printf("Bandwidth (+flush):       %8.3f GiB/s\n", total_bytes / (t_bar_flush - t_start) / GiB);
            printf("Bandwidth (+close):       %8.3f GiB/s\n", total_bytes / t_total / GiB);
            printf("===== END REPORT =====\n\n");
            fflush(stdout);
        }
    }

    free(data);

    MPI_Finalize();
    return 0;
}

