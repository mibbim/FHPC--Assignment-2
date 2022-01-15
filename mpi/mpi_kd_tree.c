#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <memory.h>
#include <math.h> //pow

#include <stdint.h>
#include <limits.h>

#if SIZE_MAX == UCHAR_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#endif

void *OOM_GUARD(void *p)
{
  if (p == NULL)
  {
    fprintf(stderr, "out of memory\n");
    exit(EXIT_FAILURE);
  }
  return p;
};

#if !defined(DOUBLE_PRECISION)
#define TYPE float
#define MPI_TYPE MPI_FLOAT
#else
#define TYPE double
#define MPI_TYPE MPI_DOUBLE
#endif

#define NDIM 2

typedef struct KdNode
{
  int axis;   // axis of the split
  size_t idx; // index in the array
  size_t left_idx, right_idx;
  TYPE value[NDIM]; // value stored in the node and splitting value
} KdNode;

void print_dataset(TYPE *dataset, size_t dataset_size, int ndim)
/* Utility fuction */
{
  // size_t effective_size = dataset_size * ndim;
  for (size_t i = 0; i < dataset_size; ++i)
  {
    size_t offset = i * ndim;
    for (int k = 0; k < ndim; ++k)
      printf(" %f ", dataset[offset + k]);
    printf("\n");
  }
};

void print_k_point(TYPE *p)
/* Utility function */
{
  for (int i = 0; i < NDIM; ++i)
  {
    // printf("p[%d]: %f,", i, p[i]);
    printf("%f, ", p[i]);
  }
  printf("\n");
  fflush(stdout);
}

void print_node(KdNode *node)
/* Utility fuction */
{
  printf("\n------------------------\n");
  printf("\tnode with index %ld at location %u\n", node->idx, node);
  printf("\t value: ");
  print_k_point(node->value);
  printf("\tAxis: %d \n\tLeft: %ld, Right: %ld \n\n", node->axis, node->left_idx, node->right_idx);
  printf("------------------------\n\n");
  fflush(stdout);
}

void swap_k(TYPE *a, TYPE *b)
{
  TYPE temp[NDIM];
  memcpy(temp, a, NDIM * sizeof(TYPE));
  memcpy(a, b, NDIM * sizeof(TYPE));
  memcpy(b, temp, NDIM * sizeof(TYPE));
}

TYPE *partition_k(TYPE *start, TYPE *end, TYPE pivot_value, int axis)
{
  TYPE *l = start;
  TYPE *r = end;

  int step = NDIM;

  TYPE *max_l = start;

  while (1)
  {
    while ((l <= r) && l[axis] <= pivot_value)
    {
      if (l[axis] > max_l[axis])
        max_l = l;
      l += step;
    }

    while ((l <= r) && r[axis] > pivot_value)
    {
      r -= step;
    }

    if (l > r)
      break;

    swap_k(l, r);

    if (l[axis] > max_l[axis])
      max_l = l;
    l += step;
    r -= step;
  }

  if (r[axis] != max_l[axis])
    swap_k(r, max_l);
  return r;
}

KdNode *build_kdtree(TYPE *dataset_start, TYPE *dataset_end, // addresses of the first and the las point in the dataset
                     KdNode *tree_location,                  // location of the root of the tree
                     int prev_axis,                          // axis uset for the partitioning at the previous branch
                     TYPE *mins, TYPE *maxs,                 // vectors of extreem values in the curren branch along each axes
                     size_t my_idx,
                     size_t *current_last_index)
/*
Note, the implementation is not ideal, the leaves point to the root
maybe the index should be shifted of an int do that root has index 1 and leaves have childs 0;
*/
{
  if (dataset_start > dataset_end)
    return 0;

  KdNode *this_node = tree_location + my_idx;
  this_node->idx = my_idx;
  this_node->axis = (prev_axis + 1) % NDIM;

  TYPE l_maxs[NDIM];
  TYPE r_mins[NDIM];

  memcpy(l_maxs, maxs, NDIM * sizeof(TYPE));
  memcpy(r_mins, mins, NDIM * sizeof(TYPE));

  TYPE mean = 0.5 * (mins[this_node->axis] + maxs[this_node->axis]);
  TYPE *pivot = partition_k(dataset_start, dataset_end, mean, this_node->axis);

  memcpy(this_node->value, pivot, NDIM * sizeof(TYPE));

  l_maxs[this_node->axis] = pivot[this_node->axis];
  r_mins[this_node->axis] = pivot[this_node->axis];

#ifdef DEBUG
  printf("Node %ld\n", this_node->idx);
  printf("pivot: %f, axes: %d = operated on: \n", pivot[this_node->axis], this_node->axis);

  print_dataset(dataset_start, (dataset_end - dataset_start + NDIM) / NDIM, NDIM);
  TYPE *L_END = pivot - NDIM;
  TYPE *R_STA = pivot + NDIM;
#endif

  if (dataset_start < pivot)
  {
    this_node->left_idx = ++(*current_last_index);
    build_kdtree(dataset_start, pivot - NDIM, tree_location, this_node->axis, mins, l_maxs, this_node->left_idx, current_last_index);
  }
  else
    this_node->left_idx = 0;

  if (pivot < dataset_end)
  {
    this_node->right_idx = ++(*current_last_index);
    build_kdtree(pivot + NDIM, dataset_end, tree_location, this_node->axis, r_mins, maxs, this_node->right_idx, current_last_index);
  }
  else
    this_node->right_idx = 0;

  return this_node;
}

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

int two_pow(int exponent)
{
  return (1 << exponent);
}

int main(int argc, char **argv)
{
  int myid, numprocs;
  int master = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  size_t dataset_size = 10;
  KdNode *local_tree;
  TYPE *dataset_start;
  TYPE *dataset_end;

  if (myid == master)
  {
    srand48(12345);
    dataset_start = create_dataset(dataset_size);
    dataset_end = dataset_start + (dataset_size * NDIM) - NDIM;
    KdNode *master_tree = (KdNode *)malloc(dataset_size * sizeof(KdNode));
    local_tree = master_tree;
  }

  TYPE mins[NDIM];
  TYPE maxs[NDIM];

  for (int i = 0; i < NDIM; ++i)
  {
    mins[i] = 0;
    maxs[i] = 1;
  }
  maxs[0] = dataset_size;
  TYPE l_maxs[NDIM];
  TYPE r_mins[NDIM];

  MPI_Status count_status;
  MPI_Status dataset_status;
  MPI_Request request_count;
  MPI_Request request;

  if (myid == master)
  {

    int level = 0;
    int axis = 0;
    KdNode *this_node = local_tree;
    this_node->axis = axis;

    TYPE mean = 0.5 * (mins[this_node->axis] + maxs[this_node->axis]);
    TYPE *pivot = partition_k(dataset_start, dataset_end, mean, this_node->axis);

    memcpy(l_maxs, maxs, NDIM * sizeof(TYPE));
    memcpy(r_mins, mins, NDIM * sizeof(TYPE));

    l_maxs[this_node->axis] = pivot[this_node->axis];
    r_mins[this_node->axis] = pivot[this_node->axis];

    ++level;
    int reciever_offset = two_pow(level - 1);

    MPI_Request request_count;
    MPI_Request request;

    if (dataset_start < pivot)
    {
      size_t r_count = pivot - dataset_start;
#ifdef DEBUG
      if (myid + reciever_offset != 1)
        printf(" ");

#endif
      // maybe not level
      // MPI_Isend(&r_count, 1, MPI_TYPE, myid + reciever_offset, level * 100, MPI_COMM_WORLD, &request_count); // sending data
      MPI_Send(&r_count, 1, my_MPI_SIZE_T, myid + 1, 0, MPI_COMM_WORLD); //, &request_count); // sending data
      // MPI_Isend(dataset_start, r_count, MPI_TYPE, myid + reciever_offset, level, MPI_COMM_WORLD, &request);
      printf("\nproc: %d, sended count %ld", myid, r_count);
      fflush(stdout);
      MPI_Isend(dataset_start, r_count, MPI_TYPE, 1, 1, MPI_COMM_WORLD, &request);
      printf("\nproc: %d, sended data", myid);
      fflush(stdout);

      // work_on first_half(++level )
#ifdef DEBUG
      printf(" ");
#endif
    }
  }

  if (myid == 1)
  {
    int level = 1;
    size_t data_count;
    int reciever_offset = two_pow(level - 1);
    MPI_Recv(&data_count, 1, my_MPI_SIZE_T, myid - reciever_offset, 0, MPI_COMM_WORLD, &count_status);
#ifdef DEBUG
    printf("\nproc: %d, recieved count %ld", myid, data_count);
    fflush(stdout);
#endif
    dataset_start = (TYPE *)OOM_GUARD(malloc(data_count * sizeof(TYPE)));
#ifdef DEBUG
    printf("\nproc: %d, allocated array", myid);
    fflush(stdout);
#endif
    MPI_Recv(dataset_start, data_count, MPI_TYPE, master, 1, MPI_COMM_WORLD, &dataset_status);
#ifdef DEBUG
    printf("\nproc: %d, recieved data", myid);
    print_dataset(dataset_start, data_count / NDIM, NDIM);
    fflush(stdout);
    printf(" ");
#endif
  }

  size_t current_last_index = 0;
  size_t starting_idx = 0;

#ifdef DEBUG
  printf("TREE CREATED \n");
  fflush(stdout);
  // treeprint(my_tree, 0);
  fflush(stdout);
#endif
  MPI_Finalize();

  return 0;
}