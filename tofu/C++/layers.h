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

        this->weights = (struct tensor *)malloc(sizeof(struct tensor));
        this->bias = (struct tensor *)malloc(sizeof(struct tensor));
			
        this->weights->rows = in_feat;
        this->weights->cols = out_feat;

        this->bias->rows = 1;
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
                    
                    result->data[i][j] = result->data[i][j] + bias->data[0][j];
    
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

    TanH(char name[]="tanh", bool trainable=false)
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

class BatchNormalization
{
    char *name;
    long double momentum;

    long double gamma;
    long double beta;
    long double epsilon;

    bool trainable = true;

    int axis;

    struct tensor *X_norm;

    long double mu, var;

    BatchNormalization(long double momentum=0.99, long double epsilon=0.00000001, int axis=0, bool trainable=true, char *name="batchnorm")
    {
        this->name = name;
        this->momentum = momentum;

        this->gamma = gamma;
        this->beta = beta;
        this->epsilon = epsilon;

        this->trainable = trainable;

        this->axis = axis;

        this->X_norm =  (struct tensor *)malloc(sizeof(struct tensor));
    }

    private:
        long double means(struct tensor *inputs, int axis=0)
        {
            long double mean = 0;
            long double sum = 0;

            for(int i=0; i<inputs->rows; i++)
            {
                for(int j=0; j<inputs->cols; j++)
                {
                    sum += inputs->data[i][j];
                }
            }

            mean = sum / (long double)(inputs->rows * inputs->cols);

            return mean;
        }

        long double variance(struct tensor *inputs, int axis=0)
        {
            long double mean = this->means(inputs);
            long double sum = 0;
            long double var = 0;

            for(int i=0; i<inputs->rows; i++)
            {
                for(int j=0; j<inputs->cols; j++)
                {
                    sum += pow((inputs->data[i][j] - mean), 2);
                }
            }

            var = sum / (long double)(inputs->rows * inputs->cols);

            return var;
        }

    public:
        struct tensor *forward(struct tensor *inputs)
        {
            if (this->trainable==false)
            {
                for(int i=0; i<inputs->rows; i++)
                {
                    for(int j=0; j<inputs->cols; j++)
                    {
                        inputs->data[i][j] = inputs->data[i][j] - this->mu;
                    }
                }

                this->X_norm = divide(inputs, sqrt((this->var + this->epsilon)));

                struct tensor *out = multiply(this->gamma, this->X_norm);

                for(int i=0; i<out->rows; i++)
                {
                    for(int j=0; j<out->cols; j++)
                    {
                        out->data[i][j] = out->data[i][j] + this->beta;
                    }
                }
			    
                return out;
            }

            this->mu = means(inputs, this->axis);
            this->var = variance(inputs, this->axis);

            for(int i=0; i<inputs->rows; i++)
            {
                for(int j=0; j<inputs->cols; j++)
                {
                    inputs->data[i][j] = inputs->data[i][j] - this->mu;
                }
            }

            this->X_norm = divide(inputs, pow(sqrt((this->var + this->epsilon)), 0.5));

            struct tensor *out = multiply(this->gamma, this->X_norm);

            for(int i=0; i<out->rows; i++)
            {
                for(int j=0; j<out->cols; j++)
                {
                    out->data[i][j] = out->data[i][j] + this->beta;
                }
            }
			    
            this->mu = this->mu * this->momentum + this->mu * (1.0 - this->momentum);
            this->var = this->var * this->momentum + this->var * (1.0 - this->momentum);

            return out;
        }

        struct tensor *backward(struct tensor *inputs, struct tensor *grad_out, long double learning_rate)
        {
            /*
            N = inputs.shape[0]

            X_mu = inputs - self.mu
            std_inv = 1.0 / np.sqrt(self.var + self.epsilon)

            dX_norm = grad_out * self.gamma
            
            d_var = np.sum(dX_norm * X_mu, axis=self.axis) * -.5 * std_inv**3
            d_mu = np.sum(dX_norm * -std_inv, axis=self.axis) + d_var * np.mean(-2. * X_mu, axis=self.axis)

            dX = (dX_norm * std_inv) + (d_var * 2 * X_mu / N) + (d_mu / N)
            
            dgamma = np.sum(grad_out * self.X_norm, axis=0)
            dbeta = np.sum(grad_out, axis=0)

            self.gamma -= learning_rate * dgamma
            self.beta -= learning_rate * dbeta

            return dX 
            */
            int N = inputs->rows;

            for(int i=0; i<inputs->rows; i++)
            {
                for(int j=0; j<inputs->cols; j++)
                {
                    inputs->data[i][j] = inputs->data[i][j] - this->mu;
                }
            }

            long double std_inv = 1.0 / sqrt(this->var + this->epsilon);

            struct tensor *dX_norm = (struct tensor *)malloc(sizeof(struct tensor));
            struct tensor *d_var = (struct tensor *)malloc(sizeof(struct tensor));
            struct tensor *d_mu = (struct tensor *)malloc(sizeof(struct tensor));

            dX_norm = multiply(this->gamma, grad_out);

            for(int i=0; i<inputs->rows; i++)
            {
                for(int j=0; j<inputs->cols; j++)
                {
                    d_var->data[i][j] = inputs->data[i][j] - this->mu;
                }
            }

            // INCOMPLETE
        }
};