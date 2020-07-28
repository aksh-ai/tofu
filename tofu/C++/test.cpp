#include <iostream>
#include <stdio.h>
#include <random>
#include "layers.h"

class Net
{
  int in_feat;
  int classes;

  public:
    Net(int in_feat=16, int classes=1)
    {
      this->in_feat = in_feat;
      this->classes = classes;
    }
    
    Tensor *forward(Tensor *X)
    {
      Linear fc1(this->in_feat, 32);
      ELU a1(0.2);
      Linear fc2(32, 64);
      ELU a2(0.2);
      Linear fc3(64, this->classes);
      Sigmoid a3; 
      
      Tensor *out = fc1.forward(X);
      out = a1.forward(out);
      out = fc2.forward(out);
      out = a2.forward(out);
      out = fc3.forward(out);
      out = a3.forward(out);

      return out;
    }
};

int main()
{
    Net net(16, 1);

    Tensor *a = (Tensor *) malloc(sizeof(Tensor));

    a->shape[0] = 1, a->shape[1] = 16;

    a = xavier_initializer(a, -0.350);

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