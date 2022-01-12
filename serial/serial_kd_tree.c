#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

#if !defined(DOUBLE_PRECISION)
#define TYPE float
#else
#define TYPE double
#endif

#define NDIM 2

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

// struct kpoint TYPE[NDIM];
//  typedef TYPE Kpoint[2];
// typedef struct Kpoint{TYPE* x;} Kpoint;

typedef struct KdNode
{
    int axis;                    // axis of the split
    size_t idx;                  // index in the array
    struct KdNode *left, *right; // children
    TYPE value[NDIM];            // value stored in the node and splitting value
} KdNode;

void print_dataset(TYPE *dataset, size_t dataset_size, int ndim)
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
{
    for (int i = 0; i < NDIM; ++i)
    {
        printf("p[%d]: %f,", i, p[i]);
    }
    printf("\n");
    fflush(stdout);
}

void print_node(KdNode *node)
{
    printf("\n------------------------\n");
    printf("\tnode with index %ld at location %u\n", node->idx, node);
    printf("\t value: ");
    print_k_point(node->value);
    printf("\tAxis: %d \n\tLeft: %u, Right: %u \n\n", node->axis, node->left, node->right);
    printf("------------------------\n\n");
    fflush(stdout);
    // printf("\t left address %p \n", node->left);
    // fflush(stdout);
    // printf("\t left idx %ld ", node.left->idx);
    // fflush(stdout);
}

void print_x_dfs(KdNode *head)
{
    if (head != NULL)
    {
        print_node(head);
        print_x_dfs(head->left);
        print_x_dfs(head->right);
    }
    else
    {
        printf("Null \n");
    }
}

size_t left_child(size_t myidx)
{
    return 2 * myidx + 1;
}

size_t right_child(size_t myidx)
{
    return 2 * myidx + 2;
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
                     size_t max_idx,
                     KdNode **current_free_block)
{
    if (dataset_start > dataset_end)
        return NULL;

    size_t my_idx = (*current_free_block) - tree_location;
    KdNode *this_node = (*current_free_block)++;
    //     find_mean_approx(dataset_size, dataset_size, mean);
    this_node->idx = my_idx;
    this_node->axis = (prev_axis + 1) % NDIM;

    // TYPE l_mins[NDIM];
    // TYPE r_maxs[NDIM];
    TYPE l_maxs[NDIM];
    TYPE r_mins[NDIM];

    memcpy(l_maxs, maxs, NDIM * sizeof(TYPE));
    memcpy(r_mins, mins, NDIM * sizeof(TYPE));
    // memcpy(r_maxs, maxs, NDIM * sizeof(TYPE));
    // memcpy(l_mins, mins, NDIM * sizeof(TYPE));

    TYPE mean = 0.5 * (mins[this_node->axis] + maxs[this_node->axis]);

    TYPE *pivot = partition_k(dataset_start, dataset_end, mean, this_node->axis);

    memcpy(this_node->value, pivot, NDIM * sizeof(TYPE));

    l_maxs[this_node->axis] = pivot[this_node->axis];
    r_mins[this_node->axis] = pivot[this_node->axis];

    // size_t left_idx = left_child(my_idx);
    // size_t right_idx = right_child(my_idx);

#ifdef DEBUG
    printf("creating node: \n");
    print_node(this_node);

#endif

    if (dataset_start < pivot)
    {
        this_node->left = build_kdtree(dataset_start, pivot - NDIM, tree_location, this_node->axis, mins, l_maxs, max_idx, current_free_block);

        if (pivot < dataset_end)
        {
            this_node->right = build_kdtree(pivot + NDIM, dataset_end, tree_location, this_node->axis, r_mins, maxs, max_idx, current_free_block);
        }
        else
            this_node->right = NULL;
    }
    else
        this_node->left = NULL;

    return this_node;
}

// int main(int argc, char **argv)
int main()
{
    size_t dataset_size = 5;
    TYPE *dataset = (TYPE *)OOM_GUARD(malloc(dataset_size * NDIM * sizeof(TYPE))); // plain array

    for (size_t i = 0; i < dataset_size; ++i)
    {
        size_t offset = NDIM * i;
        dataset[offset] = i;
        for (int k = 1; k < NDIM; ++k)
        {
            dataset[offset + k] = drand48();
            // printf(" %f ", dataset[offset + k]);
        }
        // printf("\n");
    }

#ifdef DEBUG
    printf("\n DATASET CREATED \n");
    fflush(stdout);
    print_dataset(dataset, dataset_size, NDIM);
    fflush(stdout);
#endif

    TYPE mins[NDIM];
    TYPE maxs[NDIM];

    for (int i = 0; i < NDIM; ++i)
    {
        mins[i] = 0;
        maxs[i] = 1;
    }
    maxs[0] = 4;

    // KdNode *my_tree = (KdNode *)OOM_GUARD(malloc(dataset_size * sizeof(KdNode)));
    KdNode *my_tree = (KdNode *)malloc(dataset_size * sizeof(KdNode));

    // build_kdtree_v(dataset, dataset_size, NDIM, -1, my_tree, 0);
    KdNode *free_block = my_tree;
    // KdNode *my_tree = build_kdtree(dataset, dataset + NDIM,
    //              my_tree, -1, mins, maxs, dataset_size, &free_block);
    build_kdtree(dataset, dataset + (dataset_size - 1) * NDIM,
                 my_tree, -1, mins, maxs, dataset_size, &free_block);
    free_block = NULL; 
#ifdef DEBUG
    printf("TREE CREATED \n");
    fflush(stdout);

    for (size_t i = 0; i < dataset_size; ++i)
    {
        printf("i: %ld, &tree[i]: %u\n", i, &my_tree[i]);
        print_node(&my_tree[i]);
    }

    // print_x_dfs(my_tree);
    fflush(stdout);
#endif

    free(my_tree);
    return 0;
}