#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <memory.h>
#include <math.h> //pow
#include <unistd.h>
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

const int tag_param = 0;
const int tag_mins = 100;
const int tag_maxs = 200;
const int tag_data = 300;
const int tag_tree = 400;

int int_log2(int n)
{
  int res = 0;
  while (n >>= 1)
    ++res;
  return res;
}

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

void straight_treeprint(KdNode *root, size_t tree_size)
{
  for (size_t i = 0; i < tree_size; ++i)
  {
    KdNode n = root[i];
    printf("node: ");
    print_k_point(n.value);
    printf("\t left: %ld, right: %ld ", n.left_idx, n.right_idx);
    printf("idx: %ld, real idx: %ld\n", n.idx, i);
  }
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

/* KdNode *build_tree(TYPE *dataset_start, TYPE *dataset_end,
                   TYPE *mins, TYPE *maxs, int prev_axis)
{
  // Interface for the recursive function
  size_t data_count = dataset_end - dataset_start + NDIM;
  size_t tree_size = data_count / NDIM;
  KdNode *my_tree = (KdNode *)malloc(tree_size * sizeof(KdNode));
  size_t current_last_index = 0;
  size_t starting_idx = 0;

  build_kdtree_rec(dataset_start, dataset_end,
                   my_tree, prev_axis, mins, maxs, starting_idx, &current_last_index);

  return my_tree;
} */
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

void build_mpi_tree(TYPE *dataset_start, TYPE *dataset_end,
                    TYPE *mins, TYPE *maxs, KdNode *local_tree,
                    int prev_axis, size_t idx_offset,
                    int level, int myid, int max_level,
                    size_t *last_used_index)
{
  if (dataset_start > dataset_end)
    return;

  if (level >= max_level)
  {
    /* #ifdef DEBUG
        printf("proc: %d level serial: %d, &last_used_index %u, last_used_idx: %ld \n", myid, level, last_used_index, *last_used_index);
        fflush(stdout);
        // print_dataset(dataset_start, (dataset_end - dataset_start + NDIM) / NDIM, NDIM);
    #endif */
    build_kdtree_rec(dataset_start, dataset_end, local_tree, prev_axis, mins, maxs, *last_used_index, last_used_index);
    return;
  }
  size_t data_count = (dataset_end - dataset_start + NDIM);
  size_t tree_size = data_count / NDIM;

  KdNode *this_node = local_tree + *last_used_index;
  this_node->axis = (prev_axis + 1) % NDIM;
  this_node->idx = (*last_used_index)++;

  TYPE mean = 0.5 * (mins[this_node->axis] + maxs[this_node->axis]);
  TYPE *pivot = partition_k(dataset_start, dataset_end, mean, this_node->axis);

  memcpy(this_node->value, pivot, NDIM * sizeof(TYPE));

  TYPE l_maxs[NDIM];
  TYPE r_mins[NDIM];

  memcpy(l_maxs, maxs, NDIM * sizeof(TYPE));
  memcpy(r_mins, mins, NDIM * sizeof(TYPE));

  l_maxs[this_node->axis] = pivot[this_node->axis];
  r_mins[this_node->axis] = pivot[this_node->axis];

  MPI_Request req;

  size_t r_count = dataset_end - pivot;
  size_t l_count = pivot - dataset_start;
  ++level;
  int recv_offset = two_pow(level - 1);
  /* #ifdef DEBUG
    if (myid == MASTER)
    {
      printf("process: %d, at level %d before splitting:\n", myid, level, max_level);
      straight_treeprint(local_tree, tree_size);
      printf("\n\n");
      fflush(stdout);
      // printf("process: %d, at level %d (max is %d) will process the following data:\n", myid, level, max_level);
      // print_dataset(dataset_start, (dataset_end - dataset_start + NDIM) / NDIM, NDIM);
    }
  #endif */

  size_t r_idx_offset = this_node->idx + l_count / NDIM + 1;
  if (pivot < dataset_end)
  {
    // preparing needed parameters
    size_t params[2];
    params[0] = r_count;
    params[1] = r_idx_offset;
    /* #ifdef DEBUG
        if (myid == MASTER)
        {
          printf("r_idx: %ld, last_used: %ld , this_node idx: %ld, value: %f, l_count: %ld \n", r_idx_offset, *last_used_index, this_node->idx, this_node->value[0],l_count);
          fflush(stdout);
        }
    #endif */
    this_node->right_idx = r_idx_offset;
    // sending data
    MPI_Isend(&params, 2, my_MPI_SIZE_T, myid + recv_offset, level + tag_param, MPI_COMM_WORLD, &req);
    MPI_Isend(r_mins, NDIM, MPI_TYPE, myid + recv_offset, level + tag_mins, MPI_COMM_WORLD, &req);
    MPI_Isend(maxs, NDIM, MPI_TYPE, myid + recv_offset, level + tag_maxs, MPI_COMM_WORLD, &req);
    MPI_Isend(pivot + NDIM, r_count, MPI_TYPE, myid + recv_offset, level + tag_data, MPI_COMM_WORLD, &req);

    /*     printf("\n--------------------------------------------\n");
        printf("I'm procesees %d and i'm sending %d the data at level %d\n", myid, level);
        printf("tags: %d, to: %d\n", level + tag_mins, level + tag_maxs, myid + recv_offset);
        printf("--------------------------------------------\n");
        fflush(stdout);
        printf("I'm procesees %d and i'm sending the following dataset at level: %d \n", myid, level);
        print_dataset(pivot + NDIM, r_count / NDIM, NDIM); */
  }
  else
    this_node->right_idx = 0;

  if (dataset_start < pivot)
  {
    this_node->left_idx = (*last_used_index);
    build_mpi_tree(dataset_start, pivot - NDIM,
                   mins, l_maxs, local_tree, this_node->axis,
                   0, level, myid, max_level, last_used_index);
  }
  else
    this_node->left_idx = 0;
  /* #ifdef DEBUG
    if (myid == MASTER)
    {
      printf("process: %d, at level %d after left process:\n", myid, level, max_level);
      straight_treeprint(local_tree, tree_size);
      printf("\n\n");
      fflush(stdout);
      // printf("process: %d, at level %d (max is %d) will process the following data:\n", myid, level, max_level);
      // print_dataset(dataset_start, (dataset_end - dataset_start + NDIM) / NDIM, NDIM);
    }
  #endif */

  size_t r_tree_size = r_count / NDIM;
  if (pivot < dataset_end)
  {
    KdNode *right_tree = local_tree + r_idx_offset;
    // printf("I'm proc %d and I'm trying to recieve from %d, tag: %d...", myid, myid + recv_offset, level + tag_tree);
    MPI_Recv(right_tree, r_tree_size * sizeof(KdNode), MPI_UNSIGNED_CHAR,
             myid + recv_offset, level + tag_tree,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  // printf("... done\n");
}

void add_offset(KdNode *tree, size_t tree_size, size_t offset)
{
  for (size_t i = 0; i < tree_size; ++i)
  {
    tree[i].idx += offset;
    if (tree[i].right_idx)
      tree[i].right_idx += offset;
    if (tree[i].left_idx)
      tree[i].left_idx += offset;
  }
};

int get_starting_level(int myid)
{
  return int_log2(myid) + 1;
}
/* int get_starting_level(int myid, int numprocs, int max_level)
{
  int levels[numprocs];
  int delta = 1;
  int setting_level = max_level;
  while (delta < numprocs)
  {
    for (int i = 0; i < numprocs; i += delta)
    {
      levels[i] = setting_level;
    }
    --setting_level;
    ++delta;
  }
  levels[0] = 0;

  return levels[myid];
}*/

int main(int argc, char **argv)
{
  int myid, numprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  MPI_Request req;

  size_t dataset_size = 20;
  KdNode *local_tree;
  TYPE *dataset_start;
  TYPE *dataset_end;

  int max_level = int_log2(numprocs);

  if (myid == MASTER)
  {
    srand48(12345);
    dataset_start = create_dataset(dataset_size);
    dataset_end = dataset_start + (dataset_size * NDIM) - NDIM;
    local_tree = (KdNode *)malloc(dataset_size * sizeof(KdNode));
  }

  size_t params[2]; // count, idx_offset

  MPI_Status param_status;
  MPI_Status dataset_status;
  MPI_Status tree_status;
  MPI_Request request_count;
  MPI_Request request;
  MPI_Request request_mins;
  MPI_Request request_maxs;

  if (myid == MASTER)
  { /*
     //     int level = 0;
     //     int axis = 0;

     //     KdNode *this_node = local_tree;

     //     this_node->axis = axis;
     //     this_node->idx = 0;

     //     TYPE mean = 0.5 * (mins[this_node->axis] + maxs[this_node->axis]);
     //     TYPE *pivot = partition_k(dataset_start, dataset_end, mean, this_node->axis);

     //     memcpy(this_node->value, pivot, NDIM * sizeof(TYPE));

     //     memcpy(l_maxs, maxs, NDIM * sizeof(TYPE));
     //     memcpy(r_mins, mins, NDIM * sizeof(TYPE));

     //     l_maxs[this_node->axis] = pivot[this_node->axis];
     //     r_mins[this_node->axis] = pivot[this_node->axis];

     //     ++level;
     //     int reciever_offset = two_pow(level - 1);

     //     size_t r_count = dataset_end - pivot;
     //     size_t l_count = pivot - dataset_start;

     //     if (pivot < dataset_end)
     //     { // sending data
     //       size_t r_idx_offset = l_count / NDIM + 1;
     //       params[0] = r_count;
     //       params[1] = r_idx_offset;
     //       this_node->right_idx = r_idx_offset;

     //       MPI_Isend(&params, 2, my_MPI_SIZE_T, myid + 1, level + tag_param, MPI_COMM_WORLD, &request_count);
     //       MPI_Isend(&r_mins, NDIM, MPI_TYPE, myid + 1, level + tag_mins, MPI_COMM_WORLD, &request_mins);
     //       MPI_Isend(&maxs, NDIM, MPI_TYPE, myid + 1, level + tag_maxs, MPI_COMM_WORLD, &request_maxs);
     //       MPI_Isend(pivot + NDIM, r_count, MPI_TYPE, 1, level + tag_data, MPI_COMM_WORLD, &request);
     //     }
     //     else
     //       this_node->right_idx = 0;

     //     size_t last_index;
     //     if (dataset_start < pivot)
     //     {
     //       this_node->left_idx = 1;
     //       last_index = 1;
     //       build_kdtree_rec(dataset_start, pivot - NDIM, local_tree, this_node->axis, mins, l_maxs, this_node->left_idx, &last_index);
     //     }
     //     else
     //       this_node->left_idx = 0;

     //     if (pivot < dataset_end)
     //     {
     //       size_t r_tree_size = r_count / NDIM;
     //       KdNode *right_tree = local_tree + ++last_index;

     //       MPI_Recv(right_tree, r_tree_size * sizeof(KdNode), MPI_UNSIGNED_CHAR, myid + 1, level + tag_tree, MPI_COMM_WORLD, &tree_status);
     // #ifdef DEBUG
     //       printf("\n****************SENTINEL0****************\n");
     //       straight_treeprint(local_tree, dataset_size);
     //       treeprint(local_tree, 0);
     //       fflush(stdout);
     // #endif
     //     }
     */
    TYPE mins[NDIM];
    TYPE maxs[NDIM];

    for (int i = 0; i < NDIM; ++i)
    {
      mins[i] = 0;
      maxs[i] = 1;
    }
    maxs[0] = dataset_size;

    size_t tree_size = dataset_size;
    size_t last_used_index = 0;
    int level = 0;

    build_mpi_tree(dataset_start, dataset_end,
                   mins, maxs, local_tree, -1, 0, level,
                   myid, max_level, &last_used_index);

    sleep(2);
    printf("\n\n");
    straight_treeprint(local_tree, dataset_size);
    // treeprint(local_tree, 0);
  }

  if (myid != MASTER)
  {
    int level = get_starting_level(myid);
    int reciever_offset = two_pow(level - 1);
    // printf("I'm precess %d trying to recieve data from  %d, tag: %d at level %d\n", myid, myid - reciever_offset, level + tag_param, level);
    // fflush(stdout);
    MPI_Recv(&params, 2, my_MPI_SIZE_T, myid - reciever_offset, level + tag_param, MPI_COMM_WORLD, &param_status);
    size_t data_count = params[0];
    size_t tree_size = data_count / NDIM;
    size_t idx_offset = params[1];

    dataset_start = (TYPE *)OOM_GUARD(malloc(data_count * sizeof(TYPE)));
    dataset_end = dataset_start + data_count - NDIM;
    TYPE mins[NDIM];
    TYPE maxs[NDIM];

    MPI_Recv(mins, NDIM, MPI_TYPE, myid - reciever_offset, level + tag_mins, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //, &request_mins);
    MPI_Recv(maxs, NDIM, MPI_TYPE, myid - reciever_offset, level + tag_maxs, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //, &request_mins);

    // printf("###I'm precess %d trying to recieve %ld data from  %d, tag: %d at level %d\n", myid, data_count, myid - reciever_offset, level + tag_data, level);
    MPI_Recv(dataset_start, data_count, MPI_TYPE, myid - reciever_offset, level + tag_data, MPI_COMM_WORLD, &dataset_status);
    // printf("**********SENTINEL%d**********\n", myid);
    // fflush(stdout);

    // #ifdef DEBUG
    //     printf("process: %d, at level %d (max is %d) recieved the following data:\n", myid, level, max_level);
    //     print_dataset(dataset_start, (dataset_end - dataset_start + NDIM) / NDIM, NDIM);
    // #endif
    local_tree = malloc(tree_size * sizeof(KdNode));
    size_t last_index = 0;
    size_t starting_index = 0;
    /* #ifdef DEBUG
        printf("proc: %d level: %d, &last_used_index %u, last_used_idx: %ld \n", myid, level, &last_index, last_index);
        fflush(stdout);
        // print_dataset(dataset_start, (dataset_end - dataset_start + NDIM) / NDIM, NDIM);
    #endif */

    build_mpi_tree(dataset_start, dataset_end,
                   mins, maxs, local_tree, level - 1, idx_offset, level, myid, max_level, &last_index);

#ifdef DEBUG
    printf("\n**********TREE CREATED BY PROCESS %d*************\n", myid);
    treeprint(local_tree, 0);
    add_offset(local_tree, tree_size, idx_offset);
    straight_treeprint(local_tree, tree_size);
    fflush(stdout);
#endif

    // printf("I'm precess %d trying to send the created tree to %d, tag: %d at level %d\n", myid, myid - reciever_offset, level + tag_tree, level);
    fflush(stdout);
    MPI_Isend(local_tree, tree_size * sizeof(KdNode), MPI_UNSIGNED_CHAR, myid - reciever_offset, level + tag_tree, MPI_COMM_WORLD, &req);
  }

  printf("process %d terminated \n", myid);
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);

  if (myid == 0)
  {
    printf("TREE PRODUCED: \n");
    straight_treeprint(local_tree, dataset_size);
    treeprint(local_tree, 0);
    // free(dataset);
  }

  free(local_tree);
  free(dataset_start);

  MPI_Finalize();

  return 0;
}