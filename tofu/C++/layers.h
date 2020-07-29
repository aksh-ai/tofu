#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "initializers.h"

class Module 
{
  public:
    Tensor *forward(Tensor *X)
    {
      return X;
    }

    Tensor *backward(Tensor *X, Tensor *grad_loss, long double learning_rate)
    {
      return X;
    }
};

class Linear: public Module
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

    Linear(){ /*Empty Constructor*/ }

    Linear(int in_feat, int out_feat, char const *name="linear", bool trainable=true)
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

        if(this->trainable == true) 
        {
          this->weights->grad = true;
          this->bias->grad = true;
        }
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

class BatchNormalization: public Module
{
  public:
    char const *name;

    Tensor *gamma;
    Tensor *beta;
    Tensor *mu;
    Tensor *var;

    long double momentum;
    long double epsilon;
    int axis;

    bool trainable;

    BatchNormalization(){ /*Empty Constructor*/ }

    BatchNormalization(int units, long double momentum=0.99999, long double epsilon=0.00003, int axis=0, bool training=true, char const *name="batch_norm")
    {
      this->name = name;

      this->gamma = (Tensor *)malloc(sizeof(Tensor));
      this->beta = (Tensor *)malloc(sizeof(Tensor));
      this->mu = (Tensor *)malloc(sizeof(Tensor));
      this->var = (Tensor *)malloc(sizeof(Tensor));

      this->momentum = momentum;
      this->epsilon = epsilon;

      this->axis = axis;

      this->trainable = trainable;

      this->gamma->shape[0] = units;
      this->gamma->shape[1] = 1;

      this->beta->shape[0] = units;
      this->beta->shape[1] = 1;

      this->gamma = ones(this->gamma);
      this->beta = zeros(this->beta);

      if(this->trainable == true)
      {
        this->gamma->grad = true;
        this->beta->grad = true;
      }
    }

    Tensor *forward(Tensor *X)
    {
      // Forward Propagation
      return X;
    }

    Tensor *backward(Tensor *X, Tensor *grad_loss, long double learning_rate)
    {
      // Backward propagation
      return X;
    }
};

class Sigmoid: public Module
{
  public:
      char const *name;
      bool trainable;

      Sigmoid(char const *name="sigmoid", bool trainable=true)
      {
        this->name = name;
        this->trainable = trainable;
      }

      Tensor *forward(Tensor *X)
      {
        for(size_t i=0; i<X->shape[0]; i++)
        {
          for(size_t j=0; j<X->shape[1]; j++)
          {
              X->tensor[i][j] = 1.0 / (1.0 + exp(X->tensor[i][j]));
          }
        }

        return X;
      }

      Tensor *backward(Tensor *X, Tensor *grad_loss, long double learning_rate)
      {
        X = matmul(this->forward(X), subtract(this->forward(X), 1.0));
        return X;
      }
};

class TanH: public Module
{
  public:
    char const *name;
    bool trainable;
    
    TanH(char const *name="tanh", bool trainable=true)
    {
      this->name = name;
      this->trainable = trainable;
    }

    Tensor *forward(Tensor *X)
    {
      for(size_t i=0; i<X->shape[0]; i++)
      {
        for(size_t j=0; j<X->shape[1]; j++)
        {
            X->tensor[i][j] = (exp(X->tensor[i][j]) - exp(-X->tensor[i][j])) / (exp(X->tensor[i][j]) + exp(-X->tensor[i][j]));
        }
      }

      return X;
    }

    Tensor *backward(Tensor *X, Tensor *grad_loss, long double learning_rate)
    {
      X = subtract(square(this->forward(X)), 1.0);
      return X;
    }
};

class ReLU: public Module
{
  public:
    char const *name;
    bool trainable;
    
    ReLU(char const *name="relu", bool trainable=true)
    {
      this->name = name;
      this->trainable = trainable;
    }

    Tensor *forward(Tensor *X)
    {
      for(size_t i=0; i<X->shape[0]; i++)
      {
        for(size_t j=0; j<X->shape[1]; j++)
        {
          if (X->tensor[i][j] <= 0.0)
            X->tensor[i][j] = 0.0;
        }
      }

      return X;
    }

    Tensor *backward(Tensor *X, Tensor *grad_loss, long double learning_rate)
    {
      for(size_t i=0; i<X->shape[0]; i++)
      {
        for(size_t j=0; j<X->shape[1]; j++)
        {
          if (X->tensor[i][j] <= 0.0)
            X->tensor[i][j] = 0.0;
          else
            X->tensor[i][j] = 1.0;

          X->tensor[i][j] *= grad_loss->tensor[i][j];
        }
      }

      return X;
    }
};

class LeakyReLU: public Module
{
  public:
    long double alpha;
    char const *name;
    bool trainable;
    
    LeakyReLU(long double alpha=0.3, char const *name="leaky_relu", bool trainable=true)
    {
      this->alpha = alpha;
      this->name = name;
      this->trainable = trainable;
    }

    Tensor *forward(Tensor *X)
    {
      for(size_t i=0; i<X->shape[0]; i++)
      {
        for(size_t j=0; j<X->shape[1]; j++)
        {
          if (X->tensor[i][j] <= 0.0)
            X->tensor[i][j] *= this->alpha;
        }
      }

      return X;
    }

    Tensor *backward(Tensor *X, Tensor *grad_loss, long double learning_rate)
    {
      for(size_t i=0; i<X->shape[0]; i++)
      {
        for(size_t j=0; j<X->shape[1]; j++)
        {
          if (X->tensor[i][j] <= 0.0)
            X->tensor[i][j] = this->alpha;
          else
            X->tensor[i][j] = 1.0;

          X->tensor[i][j] *= grad_loss->tensor[i][j];
        }
      }

      return X;
    }
};

class ELU: public Module
{
  public:
    long double alpha;
    char const *name;
    bool trainable;
    
    ELU(long double alpha=0.3, char const *name="elu", bool trainable=true)
    {
      this->alpha = alpha;
      this->name = name;
      this->trainable = trainable;
    }

    Tensor *forward(Tensor *X)
    {
      for(size_t i=0; i<X->shape[0]; i++)
      {
        for(size_t j=0; j<X->shape[1]; j++)
        {
          if (X->tensor[i][j] <= 0.0)
            X->tensor[i][j] = this->alpha * (exp(X->tensor[i][j]) - 1);
        }
      }

      return X;
    }

    Tensor *backward(Tensor *X, Tensor *grad_loss, long double learning_rate)
    {
      for(size_t i=0; i<X->shape[0]; i++)
      {
        for(size_t j=0; j<X->shape[1]; j++)
        {
          if (X->tensor[i][j] <= 0.0)
            X->tensor[i][j] = this->alpha * exp(X->tensor[i][j]);
          else
            X->tensor[i][j] = 1.0;

          X->tensor[i][j] *= grad_loss->tensor[i][j];
        }
      }

      return X;
    }
};