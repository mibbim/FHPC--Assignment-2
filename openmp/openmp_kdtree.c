#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <omp.h>
#include <time.h>

// #include <unistd.h>
// #include <math.h>
// #include <string.h>

#if !defined(DOUBLE_PRECISION)
#define TYPE float
#else
#define TYPE double
#endif

#if defined(_OPENMP)
#define CPU_TIME (clock_gettime(CLOCK_REALTIME, &ts), (double)ts.tv_sec + \
                                                          (double)ts.tv_nsec * 1e-9)

#define CPU_TIME_th (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &myts), (double)myts.tv_sec + \
                                                                        (double)myts.tv_nsec * 1e-9)

#else

#define CPU_TIME (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts), (double)ts.tv_sec + \
                                                                    (double)ts.tv_nsec * 1e-9)
#endif

#define NDIM 1

/**
 * out of memomy guard
 * to be used wrapping malloc
 */
void *OOM_GUARD(void *p)
{
    if (p == NULL)
    {
        fprintf(stderr, "out of memory\n");
        exit(EXIT_FAILURE);
    }
    return p;
};

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
        printf("%f, ", p[i]);
    }
    printf("\n");
    fflush(stdout);
}

void print_node(KdNode *node)
/* Utility fuction */
{
    printf("\n------------------------\n");
    printf("\tnode with index %zu at location %u\n", node->idx, node);
    printf("\t value: ");
    print_k_point(node->value);
    printf("\tAxis: %d \n\tLeft: %zu, Right: %zu \n\n", node->axis, node->left_idx, node->right_idx);
    printf("------------------------\n\n");
    fflush(stdout);
}

void recursive_print(KdNode *root, size_t index)
{
    if (index != 0)
    {
        KdNode node = root[index];
        print_node(root + index);
        recursive_print(root, node.left_idx);
        recursive_print(root, node.right_idx);
    }
    else
    {
        printf("Null \n");
    }
}

void print_x_dfs(KdNode *root)
/* Utility function */
{
    print_node(root);
    KdNode node = root[0];
    recursive_print(root, node.left_idx);
    recursive_print(root, node.right_idx);
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

#ifdef DEBUG
    // printf("Thread %d precessed Node %zu \n", omp_get_thread_num(), this_node->idx);
#endif
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

/* KdNode *build_tree_omp(TYPE *dataset_start, TYPE *dataset_end,
                       TYPE *mins, TYPE *maxs, int prev_axis)
{
    // Interface for the recursive function
    size_t data_count = dataset_end - dataset_start;
    size_t tree_size = data_count / NDIM;
    KdNode *my_tree = (KdNode *)malloc(tree_size * sizeof(KdNode));
    size_t current_last_index = 0;
    size_t starting_idx = 0;

#ifdef DEBUG
    printf("extreems point of tree:\n");
    print_k_point(mins);
    print_k_point(maxs);
    printf("operating od axis: %d\n ", (prev_axis + 1) % NDIM);
    fflush(stdout);
#endif
#pragma omp parallel

    build_kdtree_openmp(dataset_start, dataset_end,
                        my_tree, prev_axis, mins, maxs, starting_idx, &current_last_index);

    return my_tree;
} */

void straight_treeprint(KdNode *root, size_t tree_size)
{
    for (size_t i = 0; i < tree_size; ++i)
    {
        KdNode n = root[i];
        printf("node: ");
        print_k_point(n.value);
        printf("\t left: %zu, right: %zu ", n.left_idx, n.right_idx);
        printf("idx: %zu, real idx: %zu\n", n.idx, i);
    }
}
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Invalid arg number, usage \n");
        printf("build_tee.x dataset_size thread_num\n");
        return 1;
    }

    size_t dataset_size;
    sscanf(argv[1], "%zu", &dataset_size);
    int thread_num = atoi(argv[2]);
    omp_set_num_threads(thread_num);

    struct timespec ts;
    double generation_start = CPU_TIME;
    TYPE *dataset = (TYPE *)OOM_GUARD(malloc(dataset_size * NDIM * sizeof(TYPE))); // plain array

    printf("DATASET BOUNDARIES: \n %p, %p \n", dataset, dataset + dataset_size * NDIM);

    srand48(12345);

    for (size_t i = 0; i < dataset_size; ++i)
    {
        size_t offset = NDIM * i;
        for (int k = 0; k < NDIM; ++k)
        {
            dataset[offset + k] = drand48();
        }
        // dataset[offset] = i;
    }
    // print_dataset(dataset, dataset_size, NDIM);
    double generation_end = CPU_TIME;
#ifdef DEBUG
    printf("\n DATASET CREATED in %f\n", generation_end - generation_start);
    fflush(stdout);
    // print_dataset(dataset, dataset_size, NDIM);
    // fflush(stdout);
#endif

    double build_start = CPU_TIME;
    // TYPE mins[NDIM];
    // TYPE maxs[NDIM];

    TYPE *mins = malloc(NDIM * sizeof(TYPE));
    TYPE *maxs = malloc(NDIM * sizeof(TYPE));

    for (int i = 0; i < NDIM; ++i)
    {
        mins[i] = 0;
        maxs[i] = 1;
    }
    maxs[0] = dataset_size;

    // KdNode *my_tree = build_tree_omp(dataset, dataset + (dataset_size - 1) * NDIM,
    //                              mins, maxs, -1);
    size_t current_last_index = 0;
    KdNode *my_tree = malloc(dataset_size * sizeof(KdNode));
    printf("TREE BOUNDIARIES \n %p, %p\n", my_tree, my_tree + dataset_size);
    fflush(stdout);
    // firstprivate dataset_start, dataset_end,prev_axis, mins, maxs, starting idx;
    // shared current_last_index, my_tree
    int prev_axis = -1;
#pragma omp parallel shared(current_last_index) // firstprivate(my_tree, dataset, prev_axis, mins, maxs)
    {
#pragma omp single
        my_tree = build_kdtree_openmp(dataset, dataset + (dataset_size - 1) * NDIM, my_tree,
                                      prev_axis, mins, maxs, current_last_index, &current_last_index);
    }
    double build_end = CPU_TIME;
#ifdef DEBUG
    printf("TREE of size %zu CREATED in %f \n", dataset_size, build_end - build_start);
    // straight_treeprint(my_tree, dataset_size);
    // treeprint(my_tree, 0);
    // fflush(stdout);
#endif

    // free(mins);
    // free(maxs);
    free(dataset);
    free(my_tree);
    return 0;
}