#include<stdio.h>
#include<stdlib.h>
#include "linear_model.h"

int main()
{
    struct tensor *a, *b;

    a = (struct tensor *)malloc(sizeof(struct tensor));
    b = (struct tensor *)malloc(sizeof(struct tensor));

    a->rows = 4, a->cols = 4;
    b->rows = 4, b->cols = 4;

    a->data = (long double **)malloc(a->rows * a->cols * sizeof(long double));
    b->data = (long double **)malloc(b->rows * b->cols * sizeof(long double));

    for(int r=0; r<a->rows; r++)
    {
        a->data[r] = (long double *)malloc(a->cols * sizeof(long double));

        for(int c=0; c<a->cols; c++)
        {
            a->data[r][c] = (long double) (rand() % 100);
        }
    }

    for(int r=0; r<b->rows; r++)
    {
        b->data[r] = (long double *)malloc(b->cols * sizeof(long double));

        for(int c=0; c<b->cols; c++)
        {
            b->data[r][c] = (long double) (rand() % 100);
        }
    }

    struct tensor *mul = (struct tensor *)malloc(sizeof(struct tensor));
    
    mul = matmul(a, b);

    if(mul != NULL)
    {
        printf("tensor Multiplication: \n\n");

        for(int r=0; r<mul->rows; r++)
        {
            for(int c=0; c<mul->cols; c++)
            {
                printf("%.2Lf\t", mul->data[r][c]);
            }
            printf("\n");
        }
    } 

    struct tensor *dotted = (struct tensor *)malloc(sizeof(struct tensor));

    dotted = dot(a, b);

    if(dotted != NULL)
    {
        printf("\nDot Product: \n\n");

        for(int r=0; r<dotted->rows; r++)
        {
            for(int c=0; c<dotted->cols; c++)
            {
                printf("%.2Lf\t", dotted->data[r][c]);
            }
            printf("\n");
        }
    }

    return 0;
}