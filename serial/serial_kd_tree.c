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

void print_node(KdNode *node)
{
    printf("node with index %ld at location %p\n", node->idx, node);
    printf("\t xvalue %f \n", node->value[0]);
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

void build_kdtree(TYPE *dataset, size_t size, int ndim, int axis, KdNode *head_location, size_t myidx)
{
    if (size < 1)
        return;

    KdNode *this_node_p = &head_location[myidx];
    this_node_p->axis = (axis + 1) % ndim;
    this_node_p->idx = myidx;

#ifdef DEBUG
    printf("creating a node: \n");
    printf("data: %p, size: %ld, axis: %d, head_position: %p, %p, idx: %ld \n",
           dataset, size, axis, head_location, &head_location[0], myidx);
    printf("node location: %p \n", this_node_p);
#endif

    size_t left_idx = left_child(myidx);
    size_t right_idx = right_child(myidx);
    this_node_p->left = &head_location[left_idx];
    this_node_p->right = &head_location[right_idx];

    memcpy(this_node_p->value, &dataset[size / 2 * ndim], NDIM * sizeof(TYPE));

#ifdef DEBUG
    printf("children location: \n");
    printf("\tl: %p, r: %p\n", this_node_p->left, this_node_p->right);
    printf("\tvalues: %f, %f\n\n", this_node_p->value[0], this_node_p->value[1]);
    fflush(stdout);
#endif

    // this_node_p.value = dataset + (size * ndim / 2);
    //  Only for testing purposes
    TYPE *left_part = dataset;
    size_t left_size = size / 2;
    TYPE *right_part = &dataset[left_size + 1];
    size_t right_size = size - left_size - 1;

    build_kdtree(left_part, left_size, ndim, this_node_p->axis, head_location, left_idx);
    build_kdtree(right_part, right_size, ndim, this_node_p->axis, head_location, right_idx);
}

void swap(TYPE* a, TYPE* b)
{
    TYPE temp = *a;
    *a = *b;
    *b = temp;
}


TYPE *partition(TYPE *dataset, size_t dataset_size, TYPE mean) {

}

// void build_kdtree_1D(TYPE *dataset, KdNode *location, int axis,
//                      size_t index,
//                      size_t dataset_size,
//                      TYPE min, TYPE max)
// {
//     if (dataset_size < 1)
//         return;

//     TYPE mean = 0.5 * (min + max);

//     find_mean_approx(dataset_size, dataset_size, mean);

//     p_index = partition(dataset, dataset_size);

//     // left
// }
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

    // KdNode *my_tree = (KdNode *)OOM_GUARD(malloc(dataset_size * sizeof(KdNode)));
    KdNode *my_tree = (KdNode *)malloc(dataset_size * sizeof(KdNode));
    build_kdtree(dataset, dataset_size, NDIM, -1, my_tree, 0);

#ifdef DEBUG
    printf("TREE CREATED \n");
    fflush(stdout);

    for (size_t i = 0; i < dataset_size; ++i)
    {
        printf("i: %ld, &tree[i]: %p\n", i, &my_tree[i]);
        print_node(&my_tree[i]);
    }

    // print_x_dfs(my_tree);
    fflush(stdout);
#endif

    free(my_tree);
    return 0;
}