#include <iostream>
#include <stdio.h>
#include <random>
#include "layers.h"

class Net: public Module
{
  Linear fc1;
  ELU a1 = ELU(0.2);
  Linear fc2 = Linear(32, 64);
  ELU a2 = ELU(0.2);
  Linear fc3;
  Sigmoid a3;
  
  public:
    Net(int in_feat=16, int classes=1) 
    {
      this->fc1 = Linear(in_feat, 32);
      this->fc3 = Linear(64, classes);
    }

    Tensor *forward(Tensor *X)
    {      
      Tensor *out = this->fc1.forward(X);
      out = this->a1.forward(out);
      out = this->fc2.forward(out);
      out = this->a2.forward(out);
      out = this->fc3.forward(out);
      out = this->a3.forward(out);
      return out;
    }
};

int main()
{
    Net net = Net(32, 2);

    Tensor *a = (Tensor *) malloc(sizeof(Tensor));

    a->shape[0] = 1, a->shape[1] = 32;

    a = xavier_initializer(a, 0.087);

    Tensor *res = net.forward(a);

    printf("\nForward Propagation: \n\n");
  
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