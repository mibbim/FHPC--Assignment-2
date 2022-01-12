#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

#if !defined(DOUBLE_PRECISION)
#define TYPE float
#else
#define TYPE double
#endif

#define NDIM 2

void *OOM_GUARD(void *p)
{
    if (p == NULL)
    {
        fprintf(stderr, "out of memory\n");
        exit(EXIT_FAILURE);
    }
    return p;
};

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

void swap(TYPE *a, TYPE *b)
{
    TYPE temp = *a;
    *a = *b;
    *b = temp;
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
    {
        TYPE *l = start;
        TYPE *r = end;

        int step = NDIM;

        TYPE *max_l = start;

#ifdef DEBUG
        int i = 0;
#endif

        while (1)
        {
#ifdef DEBUG
            // printf("new iteration %d axis: %d, pivot: %f \n", i++, axis, pivot_value);
            // print_dataset(start, (end + NDIM - start) / NDIM, NDIM);
#endif

#ifdef DEBUG
            // printf("processing l: ");
            // fflush(stdout);
#endif
            // while ((l<=r) && (QS_COMPARE(list[l],piv) <= 0)) l++;
            while ((l <= r) && l[axis] <= pivot_value)
            {
#ifdef DEBUG
                // printf("%f ", l[axis]);
                // fflush(stdout);
#endif

                if (l[axis] > max_l[axis])
                    max_l = l;
                l += step;
            }
#ifdef DEBUG
            // printf("processing r: ");
            // fflush(stdout);
#endif
            // while ((l<=r) && (QS_COMPARE(list[r],piv)  > 0)) r--;
            while ((l <= r) && r[axis] > pivot_value)
            {
#ifdef DEBUG
                // printf("%f ", r[axis]);
                // fflush(stdout);
#endif
                r -= step;
            }

#ifdef DEBUG
            // printf("r: %f \n", *r);
            printf("l %f, r %f \n", l[axis], r[axis]);
            // fflush(stdout);
#endif

            if (l > r)
                break;

#ifdef DEBUG
                // printf("swapping %f and %f ...", l[axis], r[axis]);
                // fflush(stdout);
#endif
            swap_k(l, r);
#ifdef DEBUG
            // printf("done \n");
            // fflush(stdout);
#endif
            if (l[axis] > max_l[axis])
                max_l = l;
            l += step;
            r -= step;
        }

#ifdef DEBUG
        // printf("swapping the max :\n");
        // printf("r: %f, %f, maxl: %f, %f\n", *r, r[1], *max_l, max_l[1]);
        // fflush(stdout);
#endif
        if (r[axis] != max_l[axis])
            swap_k(r, max_l);
        return r;
    }
}

TYPE *partition(TYPE *start, TYPE *end, TYPE pivot_value)
/* If present return the position of a pivot value,
 *   the following element one otherwise
 */
{
    TYPE *l = start;
    TYPE *r = end;

    TYPE *max_l = start;

    while (1)
    {
        // while ((l<=r) && (QS_COMPARE(list[l],piv) <= 0)) l++;

        while ((l <= r) && *l <= pivot_value)
        {
            if (*l > *max_l)
                max_l = l;
            ++l;
        }
        // while ((l<=r) && (QS_COMPARE(list[r],piv)  > 0)) r--;
        while ((l <= r) && *r > pivot_value)
            --r;

        if (l > r)
            break;

#ifdef DEBUG
            // printf("swapping %f and %f\n", *l, *r);
#endif
        swap(l, r);
        if (*l > *max_l)
            max_l = l;
        ++l;
        --r;
    }

    swap(r, max_l);
    return r;
}
int main()
{
    size_t dataset_size = 2;
    size_t effective_size = dataset_size * NDIM;
    TYPE *dataset = (TYPE *)OOM_GUARD(malloc(dataset_size * NDIM * sizeof(TYPE))); // plain array

    for (size_t i = 0; i < dataset_size; ++i)
    {
        size_t offset = NDIM * i;
        // dataset[offset] = i;
        for (int k = 0; k < NDIM; ++k)
        {
            dataset[offset + k] = drand48();
            // dataset[offset + k] = (((int)offset + k) % 5);
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

    TYPE mean = 0;
    int k = 0;
    TYPE *pivot;
    pivot = partition_k(dataset, dataset + effective_size - NDIM, mean, 0);

    pivot = partition_k(dataset, dataset + effective_size - NDIM, mean, k);

#ifdef DEBUG
    printf("\n AFTER PARTITION K=%d\n", k);
    fflush(stdout);
    print_dataset(dataset, dataset_size, NDIM);
    fflush(stdout);
    // printf("pivot: %f, following: %f, prev: %f \n", *pivot, *(pivot+1), *(pivot-1));
    printf("wanted: %f, pivot: %f \n", mean, pivot[k]);
#endif

//     k = 1;
//     pivot = partition_k(dataset, dataset + effective_size - NDIM, mean, k);

// #ifdef DEBUG
//     printf("\n AFTER PARTITION K=%d\n", k);
//     fflush(stdout);
//     print_dataset(dataset, dataset_size, NDIM);
//     fflush(stdout);
//     // printf("pivot: %f, following: %f, prev: %f \n", *pivot, *(pivot+1), *(pivot-1));
//     printf("wanted: %f, pivot: %f \n", mean, pivot[k]);
// #endif

    return 0;
}