#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <memory.h>
#include <math.h> //pow
#include <unistd.h>
#include <stdint.h>
#include <limits.h>
#include <omp.h>

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

KdNode *build_kdtree_openmp(TYPE *dataset_start, TYPE *dataset_end, // addresses of the first and the las point in the dataset
                            KdNode *tree_location,                  // location of the root of the tree
                            int prev_axis,                          // axis uset for the partitioning at the previous branch
                            TYPE *mins, TYPE *maxs,                 // vectors of extreem values in the curren branch along each axes
                            size_t my_idx,
                            size_t *current_last_index)
{
    if (dataset_start > dataset_end)
    {
        printf("Error in data riecieved\n");
        exit(66);
        return NULL;
    }

    KdNode *this_node = tree_location + my_idx;
    this_node->idx = my_idx;
    this_node->axis = (prev_axis + 1) % NDIM;

    TYPE *l_maxs = malloc(NDIM * sizeof(TYPE));
    TYPE *r_mins = malloc(NDIM * sizeof(TYPE));

    memcpy(l_maxs, maxs, NDIM * sizeof(TYPE));
    memcpy(r_mins, mins, NDIM * sizeof(TYPE));

    TYPE mean = 0.5 * (mins[this_node->axis] + maxs[this_node->axis]);
    TYPE *pivot = partition_k(dataset_start, dataset_end, mean, this_node->axis);

    memcpy(this_node->value, pivot, NDIM * sizeof(TYPE));

    l_maxs[this_node->axis] = pivot[this_node->axis];
    r_mins[this_node->axis] = pivot[this_node->axis];

    if (dataset_start < pivot)
    {
#pragma omp atomic capture
        this_node->left_idx = ++(*current_last_index);

        TYPE *p_mins = &mins[0];
        TYPE *p_maxs = &l_maxs[0];
#pragma omp task shared(current_last_index, p_mins, p_maxs) firstprivate(tree_location, dataset_start, pivot, this_node)
        build_kdtree_openmp(dataset_start, pivot - NDIM, tree_location, this_node->axis, p_mins, p_maxs, this_node->left_idx, current_last_index);
    }
    else
    {
        this_node->left_idx = 0;
        free(l_maxs);
        free(mins);
    }
    if (pivot < dataset_end)
    {
#pragma omp atomic capture
        this_node->right_idx = ++(*current_last_index);

        TYPE *p_mins = &r_mins[0];
        TYPE *p_maxs = &maxs[0];

#pragma omp task shared(current_last_index, p_mins, p_maxs) firstprivate(tree_location, dataset_end, pivot, this_node)
        build_kdtree_openmp(pivot + NDIM, dataset_end, tree_location, this_node->axis, p_mins, p_maxs, this_node->right_idx, current_last_index);
    }
    else
    {
        this_node->right_idx = 0;
        free(r_mins);
        free(maxs);
    }
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

void build_mpi_tree(TYPE *dataset_start, TYPE *dataset_end,
                    TYPE *mins, TYPE *maxs, KdNode *local_tree,
                    int prev_axis,
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
        TYPE *p_mins = malloc(NDIM * sizeof(TYPE));
        TYPE *p_maxs = malloc(NDIM * sizeof(TYPE));
        memcpy(p_mins, mins, NDIM * sizeof(TYPE));
        memcpy(p_maxs, maxs, NDIM * sizeof(TYPE));
#pragma omp parallel shared(last_used_index) // firstprivate(local_tree, dataset_start, prev_axis)
        {
#pragma omp single
            local_tree = build_kdtree_openmp(dataset_start, dataset_end, local_tree,
                                             prev_axis, p_mins, p_maxs, *last_used_index, last_used_index);
        }

        return;
    }

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

    size_t r_idx_offset = this_node->idx + l_count / NDIM + 1;
    if (pivot < dataset_end)
    {
        // preparing needed parameters
        size_t params[2];
        params[0] = r_count;
        params[1] = r_idx_offset;
        this_node->right_idx = r_idx_offset;

        // sending data
        MPI_Isend(&params, 2, my_MPI_SIZE_T, myid + recv_offset, level + tag_param, MPI_COMM_WORLD, &req);
        MPI_Isend(r_mins, NDIM, MPI_TYPE, myid + recv_offset, level + tag_mins, MPI_COMM_WORLD, &req);
        MPI_Isend(maxs, NDIM, MPI_TYPE, myid + recv_offset, level + tag_maxs, MPI_COMM_WORLD, &req);
        MPI_Isend(pivot + NDIM, r_count, MPI_TYPE, myid + recv_offset, level + tag_data, MPI_COMM_WORLD, &req);
    }
    else
        this_node->right_idx = 0;

    if (dataset_start < pivot)
    {
        this_node->left_idx = (*last_used_index);
        build_mpi_tree(dataset_start, pivot - NDIM,
                       mins, l_maxs, local_tree, this_node->axis,
                       level, myid, max_level, last_used_index);
    }
    else
        this_node->left_idx = 0;

    size_t r_tree_size = r_count / NDIM;
    if (pivot < dataset_end)
    {
        KdNode *right_tree = local_tree + r_idx_offset;
        MPI_Recv(right_tree, r_tree_size * sizeof(KdNode), MPI_UNSIGNED_CHAR,
                 myid + recv_offset, level + tag_tree,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
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

int main(int argc, char **argv)
{
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == MASTER)
    {
        if (argc < 3)
        {
            printf("Invalid arg number, usage \n");
            printf("mpirun -np N ./build_tree.x dataset_size thread_num\n");
            return 1;
        }
    }

    size_t dataset_size;
    sscanf(argv[1], "%zu", &dataset_size);
    int thread_num = atoi(argv[2]);
    if (myid == MASTER)
    {
        printf("using %d thread per process\n", thread_num);
        fflush(stdout);
    }
    omp_set_num_threads(thread_num);

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

    if (myid == MASTER)
    {
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
                       mins, maxs, local_tree, -1, level,
                       myid, max_level, &last_used_index);
        free(dataset_start);
        printf("\n\n");
    }

    if (myid != MASTER)
    {
        int level = get_starting_level(myid);
        int reciever_offset = two_pow(level - 1);
        MPI_Recv(&params, 2, my_MPI_SIZE_T, myid - reciever_offset, level + tag_param, MPI_COMM_WORLD, &param_status);
        size_t data_count = params[0];
        size_t tree_size = data_count / NDIM;

        dataset_start = (TYPE *)OOM_GUARD(malloc(data_count * sizeof(TYPE)));
        dataset_end = dataset_start + data_count - NDIM;
        TYPE mins[NDIM];
        TYPE maxs[NDIM];

        MPI_Recv(mins, NDIM, MPI_TYPE, myid - reciever_offset, level + tag_mins, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //, &request_mins);
        MPI_Recv(maxs, NDIM, MPI_TYPE, myid - reciever_offset, level + tag_maxs, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //, &request_mins);

        MPI_Recv(dataset_start, data_count, MPI_TYPE, myid - reciever_offset, level + tag_data, MPI_COMM_WORLD, &dataset_status);

        local_tree = malloc(tree_size * sizeof(KdNode));
        size_t last_index = 0;

        build_mpi_tree(dataset_start, dataset_end,
                       mins, maxs, local_tree, level - 1, level, myid, max_level, &last_index);
        free(dataset_start);
#ifdef DEBUG
        printf("\n**********TREE CREATED BY PROCESS %d*************\n", myid);
        treeprint(local_tree, 0);
        add_offset(local_tree, tree_size, idx_offset);
        straight_treeprint(local_tree, tree_size);
        fflush(stdout);
#endif
        MPI_Send(local_tree, tree_size * sizeof(KdNode), MPI_UNSIGNED_CHAR, myid - reciever_offset, level + tag_tree, MPI_COMM_WORLD);
        free(local_tree);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (myid == MASTER)
    {
        printf("TREE PRODUCED: \n");
        free(local_tree);
    }

    MPI_Finalize();

    return 0;
}