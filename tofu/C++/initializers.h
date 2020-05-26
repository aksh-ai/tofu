#include <stdio.h>
#include <random>
#include "tensor.h"

using namespace std;

default_random_engine generator; 

Tensor *xavier_initializer(Tensor *A, long double limit)
{
    long double low = -limit, high = limit;

    uniform_real_distribution<long double> xavier_distribution(low, high);

    A->tensor = (long double **) malloc(sizeof(long double) * A->shape[0] * A->shape[1]);

    for(size_t axis=0; axis<A->shape[0]; axis++)
        A->tensor[axis] = (long double *) malloc(sizeof(long double) * A->shape[1]);

    for(size_t i=0; i<A->shape[0]; i++)
    {
        for(size_t j=0; j<A->shape[1]; j++)
        {
            A->tensor[i][j] = xavier_distribution(generator);
        }
    }

    xavier_distribution.reset();

    return A;
}