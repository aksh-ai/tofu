#include<stdio.h>
#include<stdlib.h>
#include<random>
#include<math.h>

struct tensor {
	int rows, cols;
	long double **data;
};

struct tensor *matmul(struct tensor *a, struct tensor *b) 
{
	int m = a->rows, n1 = a->cols;
	int n2 = b->rows, k = b->cols;

	if(n1 == n2)
	{
		int i=0, j=0, c=0;

		struct tensor *result = (struct tensor *)malloc(sizeof(struct tensor));

		result->rows = m;
		result->cols = k;

		result->data  = (long double **)malloc(result->rows * result->cols * sizeof(long double));

		for(i=0; i<result->rows; i++)
		{
			for(j=0; j<result->cols; j++)
			{
				result->data[i] = (long double *)malloc(result->cols * sizeof(long double));
			}

		}

		for(i=0; i<result->rows; i++)
		{
			for(j=0; j<result->cols; j++)
			{
				result->data[i][j] = (long double)0;

				for(c=0; c<a->cols; c++)
					*(*(result->data + i) + j) += (*(*(a->data + i) + c)) * (*(*(b->data + c) + j));
			}
		}

		return result;
	}

	else
	{
		printf("\nTensor shape mismatch: (m, n)*(n, k) -> (m, k). Tensor A's number of columns & Tensor B's number of rows are different.\n");
		return NULL;
	}
	
	return NULL;
}

struct tensor *transpose(struct tensor *a)
{
	for (int i = 0; i < a->rows; ++i)
		for (int j = 0; j < a->cols; ++j)
		{
			a->data[j][i] = a->data[i][j];
		}

	return a;    
}

struct tensor *dot(struct tensor *a, struct tensor *b)
{
	int m = a->rows, n1 = a->cols;
	int n2 = b->rows, k = b->cols;

	if((m==n2) * (n2==k))
	{
		int i=0, j=0;

		struct tensor *result = (struct tensor *)malloc(sizeof(struct tensor));

		result->rows = m;
		result->cols = k;

		result->data  = (long double **)malloc(result->rows * result->cols * sizeof(long double));

		for(i=0; i<result->rows; i++)
		{
			for(j=0; j<result->cols; j++)
			{
				result->data[i] = (long double *)malloc(result->cols * sizeof(long double));
			}

		}

		for(i=0; i<result->rows; i++)
		{
			for(j=0; j<result->cols; j++)
				result->data[i][j] = a->data[i][j] * b->data[i][j]; 
		}

		return result;
	}

	else
	{
		printf("\nTensor shape mismatch: (m, n)*(n, k) -> (m, k). Tensor A's number of columns & Tensor B's number of rows are different.\n");
		return NULL;
	}

	return NULL;
}

struct tensor *subtract(struct tensor *a, struct tensor *b)
{
	int m = a->rows, n1 = a->cols;
	int n2 = b->rows, k = b->cols;

	if(m==n2 && n1==k)
	{
		int i=0, j=0;

		struct tensor *result = (struct tensor *)malloc(sizeof(struct tensor));

		result->rows = m;
		result->cols = k;

		result->data  = (long double **)malloc(result->rows * result->cols * sizeof(long double));

		for(i=0; i<result->rows; i++)
		{
			for(j=0; j<result->cols; j++)
			{
				result->data[i] = (long double *)malloc(result->cols * sizeof(long double));
			}

		}

		for(i=0; i<result->rows; i++)
		{            
			for(j=0; j<result->cols; j++)
				result->data[i][j] = a->data[i][j] - b->data[i][j]; 
		}

		return result;
	}

	else
	{
		printf("\nTensor shape mismatch: (m, n)*(n, k) -> (m, k). Tensor A's number of columns & Tensor B's number of rows are different.\n");
		return NULL;
	}

	return NULL;
}

struct tensor *multiply(long double value, struct tensor *a)
{
	int m = a->rows, n = a->cols;

	for(int i=0; i<m; i++)
	{
		for(int j=0; j<n; j++)
		{
			a->data[i][j] = a->data[i][j] * value;
		}
	}

	return a;
}

struct tensor *square(struct tensor *a)
{
	for(int i=0; i<a->rows; i++)
	{
		for(int j=0; j<a->cols; j++)
		{
			a->data[i][j] = (long double)pow(a->data[i][j], 2);
		}
	}

	return a;
}

struct tensor *divide(struct tensor *a, long double number)
{
	for(int i=0; i<a->rows; i++)
	{
		for(int j=0; j<a->cols; j++)
		{
			a->data[i][j] /= number;
		}
	}

	return a;
}

class Linear
{
    struct tensor *weights, *bias;
    long double limit;
    char *name;
    bool trainable;

    Linear(int in_feat, int out_feat, char name[]="linear", bool trainable=true)
    {
        limit = (long double)sqrt(6.0 / (1.0 + in_feat + out_feat));
			
        this->weights->rows = in_feat;
        this->weights->cols = out_feat;

        this->bias->rows = out_feat;
        this->bias->cols = out_feat;

        std::default_random_engine generator;
        std::uniform_real_distribution<long double> xavier_distribution(-limit, limit);

        this->trainable = trainable;
        this->name = name;

        this->weights->data  = (long double **)malloc(this->weights->rows * this->weights->cols * sizeof(long double));
        this->bias->data  = (long double **)malloc(this->bias->rows * this->bias->cols * sizeof(long double));

        for(int i=0; i<this->weights->rows; i++)
        {
            for(int j=0; j<this->weights->cols; j++)
            {
                this->weights->data[i] = (long double *)malloc(this->weights->cols * sizeof(long double));
            }
        }

        for(int i=0; i<this->bias->rows; i++)
        {
            for(int j=0; j<this->bias->cols; j++)
            {
                this->bias->data[i] = (long double *)malloc(this->bias->cols * sizeof(long double));
            }
        }

        for(int i=0; i<this->weights->rows; i++)
        {
            for(int j=0; j<this->weights->cols; j++)
            {
                this->weights->data[i][j] = xavier_distribution(generator);
            }
        }

        for(int i=0; i<this->bias->rows; i++)
        {
            for(int j=0; j<this->bias->cols; j++)
            {
                this->bias->data[i][j] = xavier_distribution(generator);
            }
        }
    }

    private:
        struct tensor *slope(struct tensor *x, struct tensor *w, struct tensor *b)
        {
            struct tensor *result = (struct tensor *)malloc(sizeof(struct tensor));

			result = matmul(x, w);

            for(int i=0; i<result->rows; i++)
            {
                for(int j=0; j<result->cols; j++)
                {
                    for(int c=0; c<bias->rows; c++)
                    {
                        for(int k=0; k<bias->cols; k++)
                        {
                            result->data[i][j] = result->data[i][j] + bias->data[c][k];
                        }
                    }
                }
            }

            return result;
        }
    
    public:
        struct tensor *forward(struct tensor *X)
        {
            return this->slope(X, this->weights, this->bias);
        }

        struct tensor *backward(struct tensor *inputs, struct tensor *grad_out, long double learning_rate)
        {
            struct tensor *grad_in = dot(grad_out, transpose(this->weights));

            struct tensor *dw = dot(transpose(inputs), grad_out);
            struct tensor *db = grad_out;

            for(int i=0; i<dw->rows; i++)
            {
                for(int j=0; j<db->cols; j++)
                {
                    db->data[i][j] = db->data[i][j] / (db->rows * db->cols);
                }
            }

            this->weights = subtract(this->weights, multiply(learning_rate, dw));
			this->bias = subtract(this->bias, multiply(learning_rate, db));

            return grad_in;
        }    
};

class Sigmoid
{
    char *name;
    bool trainable;

    Sigmoid(char name[]="sigmoid", bool trainable=false)
    {
        this->name = name;
        this->trainable = trainable;
    }

    public:
        struct tensor *forward(struct tensor *X, int sign=+1)
        {
            for(int i=0; i<X->rows; i++)
            {
                for(int j=0; j<X->cols; j++)
                {
                    X->data[i][j] = 1.0 / (1.0 + exp(-X->data[i][j]*sign));
                }
            }

            return X;
        }

        struct tensor *backward(struct tensor *inputs, struct tensor *grad_out, long double learning_rate)
        {
            struct tensor *z = this->forward(inputs);
            struct tensor *neg_z = this->forward(inputs, -1);

            struct tensor *out = dot(z, neg_z);

            return dot(grad_out, out);
        }
};

class ReLU
{
    long double alpha;
    char *name;
    bool trainable;

    ReLU(long double alpha, char *name="relu", bool trainable=false)
    {
        this->alpha = alpha;
        this->name = name;
        this->trainable = trainable;
    }

    public:
        struct tensor *forward(struct tensor *X)
        {
            for(int i=0; i<X->rows; i++)
            {
                for(int j=0; j<X->cols; j++)
                {
                    if(X->data[i][j]<=(long double)0.0)
                        X->data[i][j] = 0.0;
                }
            }

            return X;
        }

        struct tensor *backward(struct tensor *inputs, struct tensor *grad_out, long double learning_rate)
        {
            for(int i=0; i<inputs->rows; i++)
            {
                for(int j=0; j<inputs->cols; j++)
                {
                    if(inputs->data[i][j]>(long double)0.0)
                        inputs->data[i][j] = (long double)1.0;
                    else
                        inputs->data[i][j] = (long double)0.0;
                }
            }

            return dot(grad_out, inputs);
        }
};

class LeakyReLU
{
    long double alpha;
    char *name;
    bool trainable;

    LeakyReLU(long double alpha, char *name="leaky_relu", bool trainable=false)
    {
        this->alpha = alpha;
        this->name = name;
        this->trainable = trainable;
    }

    public:
        struct tensor *forward(struct tensor *X)
        {
            for(int i=0; i<X->rows; i++)
            {
                for(int j=0; j<X->cols; j++)
                {
                    if(X->data[i][j]<=0.0)
                        X->data[i][j] = X->data[i][j] * this->alpha;
                }
            }

            return X;
        }

        struct tensor *backward(struct tensor *inputs, struct tensor *grad_out, long double learning_rate)
        {
            for(int i=0; i<inputs->rows; i++)
            {
                for(int j=0; j<inputs->cols; j++)
                {
                    if(inputs->data[i][j]<=(long double)0.0)
                        inputs->data[i][j] = (long double)this->alpha;
                }
            }

            return dot(grad_out, inputs);
        }
};

class ELU
{
    long double alpha;
    char *name;
    bool trainable;

    ELU(long double alpha, char *name="elu", bool trainable=false)
    {
        this->alpha = alpha;
        this->name = name;
        this->trainable = trainable;
    }

    public:
        struct tensor *forward(struct tensor *X)
        {
            for(int i=0; i<X->rows; i++)
            {
                for(int j=0; j<X->cols; j++)
                {
                    if(X->data[i][j]<=0.0)
                        X->data[i][j] = exp(X->data[i][j] - 1.0) * this->alpha;
                }
            }

            return X;
        }

        struct tensor *backward(struct tensor *inputs, struct tensor *grad_out, long double learning_rate)
        {
            for(int i=0; i<inputs->rows; i++)
            {
                for(int j=0; j<inputs->cols; j++)
                {
                    if(inputs->data[i][j]<=(long double)0.0)
                        inputs->data[i][j] = (long double)this->alpha * exp(inputs->data[i][j]);
                }
            }

            return dot(grad_out, inputs);
        }
};

class TanH
{
    char *name;
    bool trainable;

    TanH(char name[]="sigmoid", bool trainable=false)
    {
        this->name = name;
        this->trainable = trainable;
    }

    public:
        struct tensor *forward(struct tensor *X, int sign=+1)
        {
            for(int i=0; i<X->rows; i++)
            {
                for(int j=0; j<X->cols; j++)
                {
                    X->data[i][j] = (exp(X->data[i][j]) - exp(1.0 - X->data[i][j])) / (exp(X->data[i][j]) + exp(1.0 - X->data[i][j]));
                }
            }

            return X;
        }

        struct tensor *backward(struct tensor *inputs, struct tensor *grad_out, long double learning_rate)
        {
            struct tensor *z = this->forward(inputs);

            for(int i=0; i<z->rows; i++)
            {
                for(int j=0; j<z->cols; j++)
                {
                    z->data[i][j] = 1.0 - pow(z->data[i][j], 2);
                }
            }

            return dot(grad_out, z);
        }
};