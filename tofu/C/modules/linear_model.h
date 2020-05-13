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
		weights = (struct tensor *)malloc(sizeof(struct tensor));
		bias = (struct tensor *)malloc(sizeof(struct tensor));

		losses = (long double *)malloc(sizeof(long double));
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

			weights = subtract(weights, multiply(learning_rate, dw));
			bias = subtract(bias, multiply(learning_rate, db));
		}

		void initialize_xavier(long double limit)
		{
			std::default_random_engine generator;
			std::uniform_real_distribution<long double> xavier_distribution(-limit, limit);

			weights->data  = (long double **)malloc(weights->rows * weights->cols * sizeof(long double));
			bias->data  = (long double **)malloc(bias->rows * bias->cols * sizeof(long double));

			for(int i=0; i<weights->rows; i++)
			{
				for(int j=0; j<weights->cols; j++)
				{
					weights->data[i] = (long double *)malloc(weights->cols * sizeof(long double));
				}
			}

			for(int i=0; i<bias->rows; i++)
			{
				for(int j=0; j<bias->cols; j++)
				{
					bias->data[i] = (long double *)malloc(bias->cols * sizeof(long double));
				}
			}

			for(int i=0; i<weights->rows; i++)
			{
				for(int j=0; j<weights->cols; j++)
				{
					weights->data[i][j] = xavier_distribution(generator);
				}
			}

			for(int i=0; i<bias->rows; i++)
			{
				for(int j=0; j<bias->cols; j++)
				{
					bias->data[i][j] = xavier_distribution(generator);
				}
			}
		}

	public:
		long double *fit(struct tensor *X, struct tensor *y, int epochs=30, long double learning_rate=0.1, int verbose=1)
		{
			long double limit = (long double)sqrt(6.0 / (1.0 + X->rows + X->cols));
			
			weights->rows = X->cols;
			weights->cols = 1;

			bias->rows = 1;
			bias->cols =1;

			initialize_xavier(limit);

			int num_samples = X->rows;
			long double loss_val = 0;

			for(int e=0; e<epochs; e++)
			{
				loss_val = loss(X, y, weights, bias);
				optimize(X, y, weights, bias, learning_rate);

				if((e==0) || (e==(epochs-1)) || ((epochs%verbose)==0))
					printf("Epoch %d, Loss: %04.4Lf", (e+1), loss_val);

				losses[e] = loss_val;    
			}

			return losses;
		}

		long double **predict(struct tensor *X)
		{
			struct tensor *prediction = (struct tensor *)malloc(sizeof(struct tensor));

			prediction = slope(X, weights, bias);

			return prediction->data;
		}

};