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

class LinearRegression
{
    private:
        Tensor *slope(Tensor *X, Tensor *weights, Tensor *bias)
        {
            Tensor *out = matmul(X, weights);

            for(size_t i=0; i<out->shape[0]; i++)
            {
                for(size_t j=0; j<out->shape[1]; j++)
                {
                    out->tensor[i][j] += bias->tensor[0][j];
                }
            }

            return out;
        }

        Tensor *loss(Tensor *X, Tensor *y, Tensor *weights, Tensor *bias)
        {
            Tensor *y_pred = (Tensor *) malloc(sizeof(Tensor));
            y_pred = this->slope(X, weights, bias);
            return square(matsub(y_pred, y));
        }

        Tensor *grad_loss(Tensor *X, Tensor *y, Tensor *weights, Tensor *bias)
        {
            Tensor *y_pred = (Tensor *) malloc(sizeof(Tensor));
            y_pred = this->slope(X, weights, bias);
            return multiply(matsub(y_pred, y), 2.0);
        }

        Tensor *optimizer(Tensor *X, Tensor *y, Tensor *weights, Tensor *bias, long double learning_rate)
        {
            Tensor *loss_grad = grad_loss(X, y, weights, bias);

            Tensor *dw = dot(loss_grad, X);
            Tensor *db = multiply(bias, (mean(loss_grad) * (long double)X->shape[0]));

            weights = matsub(weights, multiply(dw, learning_rate));
            bias = matsub(bias, multiply(db, learning_rate));

            return weights, bias;
        }

    public:
        Tensor *weights = (Tensor *) malloc(sizeof(Tensor));
        Tensor *bias = (Tensor *) malloc(sizeof(Tensor));

        long double *losses = (long double *) malloc(sizeof(long double));

        LinearRegression()
        {

        }

        void fit(Tensor *X, Tensor *y, int batch_size=32, int epochs=30, long double learning_rate=0.1, int verbose=1)
        {
            long double limit = sqrt(6.0 / (X->shape[0] + X->shape[1] + 1.0));

            Tensor *w = (Tensor *) malloc(sizeof(Tensor));
            Tensor *b = (Tensor *) malloc(sizeof(Tensor));
            
            w->shape[0] = X->shape[1], w->shape[1] = 1;
            b->shape[0] = 1, b->shape[1] = 1;

            w = xavier_initializer(w, limit);
            b = xavier_initializer(b, limit);

            int num_samples = X->shape[0];
            long double t_loss = 0.0;

            for(int e=0; e<epochs; e++)
            {
                if((e==0) || (e==(epochs-1)) || (e%verbose == 0))
                {
                    printf("Epoch %d\n", e+1);
                    printf("Loss: %.4Lf\n", t_loss);
                }
            }

            this->weights = w;
            this->bias = b;
        }
};