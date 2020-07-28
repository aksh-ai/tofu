#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "initializers.h"

class Linear
{
    private:
        int in_feat;
        int out_feat;
        long double limit;

    public:
        char const *name;
        Tensor *weights;
        Tensor *bias;
        bool trainable;

        Linear(int in_feat, int out_feat, char const *name="linear", bool trianable=true)
        {
            this->name = name;

            this->in_feat = in_feat;
            this->out_feat = out_feat;

            this->limit = (long double) sqrt(6.0 / (this->in_feat + this->out_feat + 1.0));

            this->weights = (Tensor *) malloc(sizeof(Tensor));
            this->bias = (Tensor *) malloc(sizeof(Tensor));

            this->weights->shape[0] = this->in_feat, this->weights->shape[1] = this->out_feat;
            this->bias->shape[0] = 1, this->bias->shape[1] = this->out_feat;

            this->weights = xavier_initializer(this->weights, this->limit);
            this->bias = xavier_initializer(this->bias, this->limit);

            this->trainable = trainable;
        }

        Tensor *forward(Tensor *X)
        {
            Tensor *out = matmul(X, this->weights);

            for(size_t i=0; i<out->shape[0]; i++)
            {
                for(size_t j=0; j<out->shape[1]; j++)
                {
                    out->tensor[i][j] += this->bias->tensor[0][j];
                }
            }

            return out;
        }

        Tensor *backward(Tensor *X, Tensor *grad_loss, long double learning_rate)
        {
            Tensor *new_grad_loss = dot(grad_loss, transpose(this->weights));

            Tensor *dw = dot(grad_loss, X);
            Tensor *db = multiply(this->bias, (mean(grad_loss) * (long double)X->shape[0]));

            this->weights = matsub(this->weights, multiply(dw, learning_rate));
            this->bias = matsub(this->bias, multiply(db, learning_rate));
            
            return new_grad_loss;
        }

};