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

class LinearRegression
{
	struct tensor *weights, *bias;
	long double *losses;

	LinearRegression()
	{
		this->weights = (struct tensor *)malloc(sizeof(struct tensor));
		this->bias = (struct tensor *)malloc(sizeof(struct tensor));

		this->losses = (long double *)malloc(sizeof(long double));
	}

	private:
		struct tensor *slope(struct tensor *x, struct tensor *w, struct tensor *b)
		{
			struct tensor *result = (struct tensor *)malloc(sizeof(struct tensor));

			result = matmul(x, w);

			for(int i=0; i<result->rows; i++)
			{
				for(int j=0; j<result->cols; j++)
					*(*(result->data + i) + j) += (*(*(result->data + i) + j)) + *(*(b->data + 0));
			}

			return result;
		}

		long double loss(struct tensor *x, struct tensor *y, struct tensor *w, struct tensor *b)
		{
			long double loss = 0;

			int num_samples = x->rows;

			struct tensor *y_hat = (struct tensor *)malloc(sizeof(struct tensor));

			for(int i=0; i<w->rows; i++)
			{
				for(int j=0; j<w->cols; j++)
					y_hat->data[i] = (long double *)malloc(y_hat->cols * sizeof(long double));
			}

			y_hat = slope(x, w, b);

			struct tensor *loss_tensor = divide(square(subtract(y, y_hat)), num_samples);

			for(int i=0; i<loss_tensor->rows; i++)
			{
				for(int j=0; j<loss_tensor->rows; j++)
					loss += loss_tensor->data[i][j];
			}

			return loss;
		}

		void optimize(struct tensor *x, struct tensor *y, struct tensor *w, struct tensor *b, long double learning_rate)
		{
			struct tensor *dw = (struct tensor *)malloc(sizeof(struct tensor));
			struct tensor *db = (struct tensor *)malloc(sizeof(struct tensor));

			dw = multiply((long double)2.0, dot(transpose(subtract(y, slope(x, w, b))), x));
			db = multiply((long double)2.0, subtract(y, slope(x, w, b)));

			this->weights = subtract(this->weights, multiply(learning_rate, dw));
			this->bias = subtract(this->bias, multiply(learning_rate, db));
		}

		void initialize_xavier(long double limit)
		{
			std::default_random_engine generator;
			std::uniform_real_distribution<long double> xavier_distribution(-limit, limit);

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

	public:
		long double *fit(struct tensor *X, struct tensor *y, int epochs=30, long double learning_rate=0.1, int verbose=1)
		{
			long double limit = (long double)sqrt(6.0 / (1.0 + X->rows + X->cols));
			
			this->weights->rows = X->cols;
			this->weights->cols = 1;

			this->bias->rows = 1;
			this->bias->cols =1;

			this->initialize_xavier(limit);

			int num_samples = X->rows;
			long double loss_val = 0;

			for(int e=0; e<epochs; e++)
			{
				loss_val = this->loss(X, y, this->weights, this->bias);
				this->optimize(X, y, this->weights, this->bias, learning_rate);

				if((e==0) || (e==(epochs-1)) || ((epochs%verbose)==0))
					printf("Epoch %d, Loss: %04.4Lf", (e+1), loss_val);

				this->losses[e] = loss_val;    
			}

			return this->losses;
		}

		long double **predict(struct tensor *X)
		{
			struct tensor *prediction = (struct tensor *)malloc(sizeof(struct tensor));

			prediction = this->slope(X, this->weights, this->bias);

			return prediction->data;
		}

};