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

typedef struct KdNode
{
    int axis;   // axis of the split
    size_t idx; // index in the array
    // struct KdNode *left, *right; // children
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

    // if (root != NULL)
    // {

    //     print_x_dfs(root->left);
    //     print_x_dfs(root->right);
    // }
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
void treeprint(KdNode *root, int level){
    for (int i = 0; i < level; i++)
        printf(i == level - 1 ? " |+|" : "  ");

    KdNode node = root[0];

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
#ifdef DEBUG
    if (my_idx == 3)
        printf(" ");
#endif

    TYPE *pivot = partition_k(dataset_start, dataset_end, mean, this_node->axis);

    memcpy(this_node->value, pivot, NDIM * sizeof(TYPE));

    l_maxs[this_node->axis] = pivot[this_node->axis];
    r_mins[this_node->axis] = pivot[this_node->axis];

#ifdef DEBUG
    printf("Node %ld\n", this_node->idx);
    printf("pivot: %f, axes: %d = operated on: \n", pivot[this_node->axis], this_node->axis);
    // print_node(this_node);

    print_dataset(dataset_start, (dataset_end - dataset_start + NDIM) / NDIM, NDIM);
    TYPE *L_END = pivot - NDIM;
    TYPE *R_STA = pivot + NDIM;
#endif

    if (dataset_start < pivot)
    {
        size_t l_idx = ++(*current_last_index);
        this_node->left_idx = l_idx;
        // this_node->left =
        build_kdtree(dataset_start, pivot - NDIM, tree_location, this_node->axis, mins, l_maxs, this_node->left_idx, current_last_index);
    }
    else
        // this_node->left = NULL;
        this_node->left_idx = 0;

    if (pivot < dataset_end)
    {
        size_t r_idx = ++(*current_last_index);
        this_node->right_idx = r_idx;
        // this_node->right =
        build_kdtree(pivot + NDIM, dataset_end, tree_location, this_node->axis, r_mins, maxs, this_node->right_idx, current_last_index);
    }
    else
        // this_node->right = NULL;
        this_node->right_idx = 0;

    return this_node;
}

// int main(int argc, char **argv)
int main()
{
    size_t dataset_size = 10;
    TYPE *dataset = (TYPE *)OOM_GUARD(malloc(dataset_size * NDIM * sizeof(TYPE))); // plain array

    srand48(12345);

    for (size_t i = 0; i < dataset_size; ++i)
    {
        size_t offset = NDIM * i;
        for (int k = 0; k < NDIM; ++k)
        {
            dataset[offset + k] = drand48();
            // printf(" %f ", dataset[offset + k]);
        }
        dataset[offset] = i;
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
    maxs[0] = dataset_size;

    // KdNode *my_tree = (KdNode *)OOM_GUARD(malloc(dataset_size * sizeof(KdNode)));
    KdNode *my_tree = (KdNode *)malloc(dataset_size * sizeof(KdNode));

    // build_kdtree_v(dataset, dataset_size, NDIM, -1, my_tree, 0);
    KdNode *free_block = my_tree;
    // KdNode *my_tree = build_kdtree(dataset, dataset + NDIM,
    //              my_tree, -1, mins, maxs, dataset_size, &free_block);
    size_t current_last_index = 0;
    size_t starting_idx = 0;
    build_kdtree(dataset, dataset + (dataset_size - 1) * NDIM,
                 my_tree, -1, mins, maxs, starting_idx, &current_last_index);
    free_block = NULL;
#ifdef DEBUG
    printf("TREE CREATED \n");
    fflush(stdout);

    // for (size_t i = 0; i < dataset_size; ++i)
    // {
    //     printf("i: %ld, &tree[i]: %u\n", i, &my_tree[i]);
    //     print_node(&my_tree[i]);
    // }

    // print_x_dfs(my_tree);
    fflush(stdout);
#endif

    treeprint(my_tree, 0);

    free(my_tree);
    return 0;
}