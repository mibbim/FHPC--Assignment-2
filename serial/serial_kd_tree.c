#include <stdlib.h>
#include <stdio.h>

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
typedef TYPE Kpoint[NDIM];

typedef struct KdNode
{
    int axis;
    Kpoint split;
    struct KdNode *left, *right;
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

// int main(int argc, char **argv)
int main()
{
    size_t dataset_size = 5;
    TYPE *dataset = (TYPE *)OOM_GUARD(malloc(dataset_size * NDIM * sizeof(TYPE)));

    for (size_t i = 0; i < dataset_size; ++i)
    {
        size_t offset = NDIM * i;
        for (int k = 0; k < NDIM; ++k)
        {
            dataset[offset + k] = drand48();
            // printf(" %f ", dataset[offset + k]);
        }
        // printf("\n");
    }

#ifdef DEBUG
    print_dataset(dataset, dataset_size, NDIM);
#endif

    return 0;
}