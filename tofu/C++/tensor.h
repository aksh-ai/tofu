#include<stdio.h>
#include<math.h>
#include<stdlib.h>

struct Tensor
{
    int shape[2];
    long double **tensor;
};

Tensor *matmul(Tensor *A, Tensor *B)
{
    if(A->shape[1] == B->shape[0])
    {
        Tensor *C = (Tensor *) malloc(sizeof(Tensor));

        C->shape[0] = A->shape[0];
        C->shape[1] = B->shape[1];

        C->tensor = (long double **) malloc(sizeof(long double) * C->shape[0] * C->shape[1]);

        for(size_t axis=0; axis<C->shape[0]; axis++)
            C->tensor[axis] = (long double *) malloc(sizeof(long double) * C->shape[1]);

        for(size_t i=0; i<C->shape[0]; i++)
        {
            for(size_t j=0; j<C->shape[1]; j++)
            {
                C->tensor[i][j] = 0.0;

                for(int k=0; k<A->shape[1]; k++)
                    C->tensor[i][j] += A->tensor[i][k] * B->tensor[k][j];
            
            }
        }    

        return C;
    }

    else
    {
        printf("\nTensor shape mismatch: (m, n) * (n, k) -> (m, k). Tensor A's number of columns & Tensor B's number of rows are different.\n");
		return NULL;
    }
    
    return NULL;
}

Tensor *matadd(Tensor *A, Tensor*B)
{
    if((A->shape[0] == B->shape[0]) && (A->shape[1] == B->shape[1]))
    {
        Tensor *C = (Tensor *) malloc(sizeof(Tensor));

        C->shape[0] = A->shape[0];
        C->shape[1] = A->shape[1];

        C->tensor = (long double **) malloc(sizeof(long double) * C->shape[0] * C->shape[1]);

        for(size_t axis=0; axis<C->shape[0]; axis++)
            C->tensor[axis] = (long double *) malloc(sizeof(long double) * C->shape[1]);

        for(size_t i=0; i<C->shape[0]; i++)
        {
            for(size_t j=0; j<C->shape[1]; j++)
            {
                C->tensor[i][j] = 0.0;
                C->tensor[i][j] = A->tensor[i][j] + B->tensor[i][j];
            }
        }    

        return C;
    }    

    else
    {
        printf("\nTensor shape mismatch: (m, n) + (m, n) -> (m, n). Tensor A & Tensor B are of different shapes.\n");
		return NULL;
    }

    return NULL;
}

Tensor *matsub(Tensor *A, Tensor*B)
{
    if((A->shape[0] == B->shape[0]) && (A->shape[1] == B->shape[1]))
    {
        Tensor *C = (Tensor *) malloc(sizeof(Tensor));

        C->shape[0] = A->shape[0];
        C->shape[1] = A->shape[1];

        C->tensor = (long double **) malloc(sizeof(long double) * C->shape[0] * C->shape[1]);

        for(size_t axis=0; axis<C->shape[0]; axis++)
            C->tensor[axis] = (long double *) malloc(sizeof(long double) * C->shape[1]);

        for(size_t i=0; i<C->shape[0]; i++)
        {
            for(size_t j=0; j<C->shape[1]; j++)
            {
                C->tensor[i][j] = 0.0;
                C->tensor[i][j] = A->tensor[i][j] - B->tensor[i][j];
            }
        }    

        return C;
    }    

    else
    {
        printf("\nTensor shape mismatch: (m, n) - (m, n) -> (m, n). Tensor A & Tensor B are of different shapes.\n");
		return NULL;
    }

    return NULL;
}

Tensor *dot(Tensor *A, Tensor*B)
{
    if((A->shape[0] == B->shape[0]) && (A->shape[1] == B->shape[1]))
    {
        Tensor *C = (Tensor *) malloc(sizeof(Tensor));

        C->shape[0] = A->shape[0];
        C->shape[1] = A->shape[1];

        C->tensor = (long double **) malloc(sizeof(long double) * C->shape[0] * C->shape[1]);

        for(size_t axis=0; axis<C->shape[0]; axis++)
            C->tensor[axis] = (long double *) malloc(sizeof(long double) * C->shape[1]);

        for(size_t i=0; i<C->shape[0]; i++)
        {
            for(size_t j=0; j<C->shape[1]; j++)
            {
                C->tensor[i][j] = 0.0;
                C->tensor[i][j] = A->tensor[i][j] * B->tensor[i][j];
            }
        }    

        return C;
    }    

    else
    {
        printf("\nTensor shape mismatch: (m, n) * (m, n) -> (m, n). Tensor A & Tensor B are of different shapes.\n");
		return NULL;
    }

    return NULL;
}

Tensor *transpose(Tensor *A)
{
    for(size_t i=0; i<A->shape[0]; i++)
        {
            for(size_t j=0; j<A->shape[1]; j++)
            {
                A->tensor[j][i] = A->tensor[i][j];
            }
        }

    return A;    
}

Tensor *add(Tensor *A, long double value)
{
    for(size_t i=0; i<A->shape[0]; i++)
        {
            for(size_t j=0; j<A->shape[1]; j++)
            {
                A->tensor[i][j] = A->tensor[i][j] + value;
            }
        }

    return A; 
}

Tensor *subtract(Tensor *A, long double value)
{
    for(size_t i=0; i<A->shape[0]; i++)
        {
            for(size_t j=0; j<A->shape[1]; j++)
            {
                A->tensor[i][j] = A->tensor[i][j] - value;
            }
        }

    return A; 
}

Tensor *multiply(Tensor *A, long double value)
{
    for(size_t i=0; i<A->shape[0]; i++)
        {
            for(size_t j=0; j<A->shape[1]; j++)
            {
                A->tensor[i][j] = A->tensor[i][j] * value;
            }
        }

    return A; 
}

Tensor *divide(Tensor *A, long double value)
{
    for(size_t i=0; i<A->shape[0]; i++)
        {
            for(size_t j=0; j<A->shape[1]; j++)
            {
                A->tensor[i][j] = A->tensor[i][j] / value;
            }
        }

    return A; 
}

Tensor *square(Tensor *A, long double value)
{
    for(size_t i=0; i<A->shape[0]; i++)
        {
            for(size_t j=0; j<A->shape[1]; j++)
            {
                A->tensor[i][j] = A->tensor[i][j] * A->tensor[i][j];
            }
        }

    return A; 
}

Tensor *sqrt(Tensor *A, long double value)
{
    for(size_t i=0; i<A->shape[0]; i++)
        {
            for(size_t j=0; j<A->shape[1]; j++)
            {
                A->tensor[i][j] = sqrt(A->tensor[i][j]);
            }
        }

    return A; 
}

long double mean(Tensor *A)
{
    long double sum;

    for(size_t i=0; i<A->shape[0]; i++)
    {
        for(size_t j=0; j<A->shape[1]; j++)
        {
            sum += A->tensor[i][j];
        }
    }

    return sum/(long double)(A->shape[0] * A->shape[1]);
}