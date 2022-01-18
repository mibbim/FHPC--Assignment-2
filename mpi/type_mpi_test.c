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

#define MASTER 0

typedef struct KdNode
{
  int axis;   // axis of the split
  size_t idx; // index in the array
  size_t left_idx;
  size_t right_idx;
  TYPE value[NDIM]; // value stored in the node and splitting value
} KdNode;

void create_mpi_kdNode(MPI_Datatype *MpiKdNode)
{
  int node_members_num = 5;
  MPI_Aint displacements[node_members_num];
  int lenghts[5] = {1, 1, 1, 1, NDIM};
  KdNode dummy_node;
  MPI_Aint base_address;

  MPI_Get_address(&dummy_node, &base_address);
  MPI_Get_address(&dummy_node.axis, &displacements[0]);
  MPI_Get_address(&dummy_node.idx, &displacements[1]);
  MPI_Get_address(&dummy_node.left_idx, &displacements[2]);
  MPI_Get_address(&dummy_node.right_idx, &displacements[3]);
  MPI_Get_address(&dummy_node.value, &displacements[4]);

  for (int i = 0; i < node_members_num; ++i)
    displacements[i] = MPI_Aint_diff(displacements[i], base_address);

  MPI_Datatype types[5] = {MPI_INT, my_MPI_SIZE_T, my_MPI_SIZE_T, my_MPI_SIZE_T, MPI_TYPE};

  MPI_Type_create_struct(node_members_num, lenghts, displacements, types, MpiKdNode);

  // return MpiKdNode;
}

MPI_Datatype MpiKdNode;

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

void recursive_treeprint(KdNode *root, size_t idx, int level)
{
  if (idx == 0)
    return;
  for (int i = 0; i < level; i++)
    printf(i == level - 1 ? " |+|" : "  ");
  KdNode node = root[idx];
  print_k_point(node.value);
  recursive_treeprint(root, node.left_idx, level + 1);
  recursive_treeprint(root, node.right_idx, level + 1);
}

void treeprint(KdNode *root, int level)
{
  for (int i = 0; i < level; i++)
    printf(i == level - 1 ? " |+|" : "  ");

  KdNode node = root[0];
  print_k_point(node.value);

  recursive_treeprint(root, node.left_idx, level + 1);
  recursive_treeprint(root, node.right_idx, level + 1);
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

  if (r > start)
  {
    if (r[axis] != max_l[axis])
      swap_k(r, max_l);
    return r;
  }
  return start;
}

void build_kdtree_rec(TYPE *dataset_start, TYPE *dataset_end, // addresses of the first and the las point in the dataset
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
    return;

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
  // printf("Node %ld\n", this_node->idx);
  // printf("pivot: %f, axes: %d = operated on: \n", pivot[this_node->axis], this_node->axis);

  // print_dataset(dataset_start, (dataset_end - dataset_start + NDIM) / NDIM, NDIM);
  TYPE *L_END = pivot - NDIM;
  TYPE *R_STA = pivot + NDIM;
#endif

  if (dataset_start < pivot)
  {
    this_node->left_idx = ++(*current_last_index);
    build_kdtree_rec(dataset_start, pivot - NDIM, tree_location, this_node->axis, mins, l_maxs, this_node->left_idx, current_last_index);
  }
  else
    this_node->left_idx = 0;

  if (pivot < dataset_end)
  {
    this_node->right_idx = ++(*current_last_index);
    build_kdtree_rec(pivot + NDIM, dataset_end, tree_location, this_node->axis, r_mins, maxs, this_node->right_idx, current_last_index);
  }
  else
    this_node->right_idx = 0;

  return;
}

KdNode *build_tree(TYPE *dataset_start, TYPE *dataset_end,
                   TYPE *mins, TYPE *maxs, int prev_axis)
{
  // Interface for the recursive function
  size_t data_count = dataset_end - dataset_start + NDIM;
  size_t node_count = data_count / NDIM;
  KdNode *my_tree = (KdNode *)malloc(node_count * sizeof(KdNode));
  size_t current_last_index = 0;
  size_t starting_idx = 0;

#ifdef DEBUG
  printf("extreems point of tree:\n");
  print_k_point(mins);
  print_k_point(maxs);
  printf("operating od axis: %d\n ", (prev_axis + 1) % NDIM);
  fflush(stdout);
#endif

  build_kdtree_rec(dataset_start, dataset_end,
                   my_tree, prev_axis, mins, maxs, starting_idx, &current_last_index);

#ifdef DEBUG
  // treeprint(my_tree, 0);
#endif

  return my_tree;
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

  return dataset;
}

int main(int argc, char **argv)
{
  int myid, numprocs;
  MPI_Init(&argc, &argv);

  // create_mpi_kdNode(&MpiKdNode);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // int node_members_num = 5;
  // MPI_Aint displacements[node_members_num];
  // int lenghts[5] = {1, 1, 1, 1, NDIM};
  // KdNode dummy_node;
  // MPI_Aint base_address;

  // MPI_Get_address(&dummy_node, &base_address);
  // MPI_Get_address(&dummy_node.axis, &displacements[0]);
  // MPI_Get_address(&dummy_node.idx, &displacements[1]);
  // MPI_Get_address(&dummy_node.left_idx, &displacements[2]);
  // MPI_Get_address(&dummy_node.right_idx, &displacements[3]);
  // MPI_Get_address(&dummy_node.value, &displacements[4]);

  // for (int i = 0; i < node_members_num; ++i)
  //   displacements[i] = MPI_Aint_diff(displacements[i], base_address);

  // MPI_Datatype types[5] = {MPI_INT, my_MPI_SIZE_T, my_MPI_SIZE_T, my_MPI_SIZE_T, MPI_TYPE};
  MPI_Status status;

  size_t dataset_size = 10;

  KdNode *local_tree;
  TYPE *dataset_start;
  TYPE *dataset_end;

  TYPE mins[NDIM];
  TYPE maxs[NDIM];

  size_t data_count = dataset_size * NDIM;
  size_t tree_size = dataset_size;
  if (myid == MASTER)
  {
    srand48(12345);
    dataset_start = create_dataset(dataset_size);
    dataset_end = dataset_start + (dataset_size * NDIM) - NDIM;
    // KdNode *local_tree; //= (KdNode *)malloc(dataset_size * sizeof(KdNode));
    for (int i = 0; i < NDIM; ++i)
    {
      mins[i] = 0;
      maxs[i] = 1;
    }
    maxs[0] = dataset_size;

    local_tree = build_tree(dataset_start, dataset_end, mins, maxs, -1);
    printf("****************In Master process****************");
    treeprint(local_tree, 0);

    MPI_Send(local_tree, tree_size * sizeof(KdNode), MPI_UNSIGNED_CHAR, 1, 1, MPI_COMM_WORLD);
  }
  if (myid == 1)
  {
    local_tree = malloc(dataset_size * sizeof(KdNode));
    MPI_Recv(local_tree, dataset_size * sizeof(KdNode), MPI_UNSIGNED_CHAR, MASTER, 1, MPI_COMM_WORLD, &status);
    printf("****************In process 1 ****************");
    treeprint(local_tree, 0);
  }

  MPI_Finalize();

  return 0;
}
