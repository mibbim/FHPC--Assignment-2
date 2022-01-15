#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#if !defined(DOUBLE_PRECISION)
#define TYPE float
#define MPI_TYPE MPI_FLOAT
#else
#define TYPE double
#define MPI_TYPE MPI_DOUBLE
#endif

#define NDIM 2

void *OOM_GUARD(void *p)
{
  if (p == NULL)
  {
    fprintf(stderr, "out of memory\n");
    exit(EXIT_FAILURE);
  }
  return p;
};

TYPE *create_dataset(size_t dataset_size)

{
  TYPE *dataset = (TYPE *)OOM_GUARD(malloc(dataset_size * NDIM * sizeof(TYPE))); // plain array

  for (size_t i = 0; i < dataset_size; ++i)
  {
    size_t offset = NDIM * i;
    for (int k = 0; k < NDIM; ++k)
    {
      dataset[offset + k] = drand48();
    }
    dataset[offset] = i;
  }

#ifdef DEBUG
  printf("\n DATASET CREATED \n");
  fflush(stdout);
  print_dataset(dataset, dataset_size, NDIM);
  fflush(stdout);
#endif
  return dataset;
}

int main(int argc, char **argv)
{

  int myid, numprocs;
  int master = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  size_t dataset_size = 10;
  TYPE *dataset_start;
  TYPE *dataset_end;

  if (myid == master)
  {
    srand48(12345);
    dataset_start = create_dataset(dataset_size);
    dataset_end = dataset_start + (dataset_size * NDIM) - NDIM;

    TYPE *pivot = dataset_start + dataset_size / 2;
    if (dataset_start < pivot)
    {
      size_t r_count = pivot - dataset_start;
      MPI_Send(&r_count, 1, MPI_TYPE, myid + 1, 0, MPI_COMM_WORLD);
#ifdef DEBUG
      printf("\nproc: %d, sended count", myid);
      fflush(stdout);
#endif
      MPI_Isend(dataset_start, r_count, MPI_TYPE, myid, 1, MPI_COMM_WORLD, &request);
#ifdef DEBUG
      printf("\nproc: %d, sended data", myid);
      fflush(stdout);
#endif
    }
  }

  if (myid == 1)
  {
    size_t dataset_count;
    MPI_Recv(&dataset_count, 1, my_MPI_SIZE_T, myid - 1, 0, MPI_COMM_WORLD, &count_status);
    dataset_start = (TYPE *)OOM_GUARD(malloc(dataset_count * sizeof(TYPE)));
  }
}
