#include <iostream>
#include <stdio.h>
#include <random>
#include "layers.h"

int main()
{
    Linear fc1(5, 2);

    Tensor *a = (Tensor *) malloc(sizeof(Tensor));
    Tensor *res = (Tensor *) malloc(sizeof(Tensor));

    a->shape[0] = 2, a->shape[1] = 5;

    a = xavier_initializer(a, 5.0);

    printf("\nLinear Layer -> Weights: \n\n");

    for(size_t i=0; i < fc1.weights->shape[0]; i++)
    {
        for (size_t j = 0; j < fc1.weights->shape[1]; j++)
        {
           printf("%Lf  ", fc1.weights->tensor[i][j]);
        }
        
        printf("\n");
    }

    printf("\nLinear Layer -> Bias: \n\n");

    for(size_t i=0; i < fc1.bias->shape[0]; i++)
    {
        for (size_t j = 0; j < fc1.bias->shape[1]; j++)
        {
           printf("%Lf  ", fc1.bias->tensor[i][j]);
        }
        
        printf("\n");
    }

    printf("\nForward Propagation: \n\n");

    res = fc1.forward(a);

    for(size_t i=0; i < res->shape[0]; i++)
    {
        for (size_t j = 0; j < res->shape[1]; j++)
        {
           printf("%Lf  ", res->tensor[i][j]);
        }
        
        printf("\n");
    }

    printf("\n");

    return 0;
}