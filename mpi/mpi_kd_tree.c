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

// void build_mpi_tree(TYPE *dataset_start, TYPE *dataset_end,
//                     TYPE *mins, TYPE *maxs,
//                     int prev_axis,
//                     int level, int myid, int max_level)
// {
//   if (dataset_start > dataset_end)
//     return;

//   size_t data_count = (dataset_end - dataset_start + NDIM);
//   size_t tree_size = data_count / NDIM;

//   if (level == max_level)
//   {
//     int lvl_offset = two_pow(level - 1);

//     KdNode *local_tree = malloc(tree_size * sizeof(KdNode));
//     build_kdtree_rec(dataset_start, dataset_end, local_tree, prev_axis, mins, maxs, 0, 0);
//     // update indexes
//     // send the tree to ancestors
//     return;
//   }

//   int recv_offset = two_pow(level - 1);
// }

void add_offset(KdNode *tree, size_t tree_size, size_t offset)
{
  for (size_t i = 0; i < tree_size; ++i)
  {
    tree[i].idx += offset;
  }
};

int main(int argc, char **argv)
{
  int myid, numprocs;
  MPI_Init(&argc, &argv);

  create_mpi_kdNode(&MpiKdNode);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  size_t dataset_size = 10;
  KdNode *local_tree;
  TYPE *dataset_start;
  TYPE *dataset_end;

  int tag_param = 0;
  int tag_mins = 100;
  int tag_maxs = 200;
  int tag_data = 300;
  int tag_tree = 400;

  TYPE mins[NDIM];
  TYPE maxs[NDIM];

  if (myid == MASTER)
  {
    srand48(12345);
    dataset_start = create_dataset(dataset_size);
    dataset_end = dataset_start + (dataset_size * NDIM) - NDIM;
    KdNode *master_tree = (KdNode *)malloc(dataset_size * sizeof(KdNode));

    local_tree = master_tree;
#ifdef DEBUG
    printf("master tree %u\n", local_tree);
    printf("local tree %u\n", master_tree);
    fflush(stdout);
#endif
  }

  // contiguous for message passing
  // TYPE extreems[2 * NDIM];
  // TYPE *mins = extreems;
  // TYPE *maxs = extreems + NDIM;

  for (int i = 0; i < NDIM; ++i)
  {
    mins[i] = 0;
    maxs[i] = 1;
  }
  maxs[0] = dataset_size;
  TYPE l_maxs[NDIM];
  TYPE r_mins[NDIM];

  size_t params[2]; // count, idx_offset

  MPI_Status param_status;
  MPI_Status dataset_status;
  MPI_Status tree_status;
  MPI_Request request_count;
  MPI_Request request;
  MPI_Request request_mins;
  MPI_Request request_maxs;

  if (myid == MASTER)
  {
    // build_tree(dataset_start, dataset_end, mins, maxs);

    int level = 0;
    int axis = 0;
    KdNode *this_node = local_tree;
    this_node->axis = axis;
    this_node->idx = 0;

    this_node->left_idx = 1;
    this_node->right_idx = 0;

    TYPE mean = 0.5 * (mins[this_node->axis] + maxs[this_node->axis]);
    TYPE *pivot = partition_k(dataset_start, dataset_end, mean, this_node->axis);

    memcpy(this_node->value, pivot, NDIM * sizeof(TYPE));

    memcpy(l_maxs, maxs, NDIM * sizeof(TYPE));
    memcpy(r_mins, mins, NDIM * sizeof(TYPE));

    l_maxs[this_node->axis] = pivot[this_node->axis];
    r_mins[this_node->axis] = pivot[this_node->axis];

    ++level;
    int reciever_offset = two_pow(level - 1);

    size_t r_count = dataset_end - pivot;
    size_t l_count = pivot - dataset_start;

    if (pivot < dataset_end)
    { // sending data
      size_t r_idx_offset = l_count;
      params[0] = r_count;
      params[1] = r_idx_offset;

      MPI_Isend(&params, 2, my_MPI_SIZE_T, myid + 1, level + tag_param, MPI_COMM_WORLD, &request_count);
      MPI_Isend(&r_mins, NDIM, MPI_TYPE, myid + 1, level + tag_mins, MPI_COMM_WORLD, &request_mins);
      MPI_Isend(&maxs, NDIM, MPI_TYPE, myid + 1, level + tag_maxs, MPI_COMM_WORLD, &request_maxs);
      MPI_Isend(pivot + NDIM, r_count, MPI_TYPE, 1, level + tag_data, MPI_COMM_WORLD, &request);
    }

    if (dataset_start < pivot)
    {
#ifdef DEBUG
      printf("\n****************SENTINEL****************\n");
      printf("local tree %u\n", local_tree);
      fflush(stdout);
#endif
      // build_tree(dataset_start, pivot - NDIM, mins, l_maxs, 0);
      // build_kdtree_rec(dataset_start, pivot - NDIM, local_tree, level - 1, mins, l_maxs, 0, 0);
      size_t last_index = 1; 
      build_kdtree_rec(dataset_start, pivot - NDIM, local_tree, this_node->axis, mins, l_maxs, this_node->left_idx, &last_index);
#ifdef DEBUG
      printf("\n****************SENTINEL****************\n");
      fflush(stdout);
#endif

      treeprint(local_tree, 0);
    }

    if (pivot < dataset_end)
    {
      MPI_Recv(local_tree + r_count + NDIM, r_count, MpiKdNode, myid + 1, level + tag_tree, MPI_COMM_WORLD, &tree_status);
    }
    treeprint(local_tree, 0);
  }

  if (myid == 1)
  {
    int level = 1;
    int reciever_offset = two_pow(level - 1);
    MPI_Recv(&params, 2, my_MPI_SIZE_T, myid - reciever_offset, level + tag_param, MPI_COMM_WORLD, &param_status);
    size_t data_count = params[0];
    size_t tree_size = data_count / NDIM;
    size_t idx_offset = params[1];

    dataset_start = (TYPE *)OOM_GUARD(malloc(data_count * sizeof(TYPE)));
    MPI_Irecv(&mins, NDIM, MPI_TYPE, myid - reciever_offset, level + tag_mins, MPI_COMM_WORLD, &request_mins);
    MPI_Irecv(&maxs, NDIM, MPI_TYPE, myid - reciever_offset, level + tag_maxs, MPI_COMM_WORLD, &request_mins);

    MPI_Recv(dataset_start, data_count, MPI_TYPE, myid - reciever_offset, level + tag_data, MPI_COMM_WORLD, &dataset_status);
    // build_tree(dataset_start, dataset_start + data_count - NDIM, mins, maxs, 2);
    local_tree = malloc(tree_size * sizeof(KdNode));
    build_kdtree_rec(dataset_start, dataset_end, local_tree,
                     level - 1, mins, maxs, 0, 0);

    add_offset(local_tree, tree_size, idx_offset);
    MPI_Send(local_tree, tree_size, MpiKdNode, myid - reciever_offset, level + tag_tree, MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}