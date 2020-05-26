#include "tensor.h"

Tensor *mae(Tensor *y_pred, Tensor *y_true)
{
    Tensor *loss = (Tensor *) malloc(sizeof(Tensor));
    loss = matsub(y_pred, y_true);
    return loss;
}

Tensor *mse(Tensor *y_pred, Tensor *y_true)
{
    Tensor *loss = (Tensor *) malloc(sizeof(Tensor));
    loss = square(matsub(y_pred, y_true));
    return loss;
}

Tensor *rmse(Tensor *y_pred, Tensor *y_true)
{
    Tensor *loss = (Tensor *) malloc(sizeof(Tensor));
    loss = sqroot(square(matsub(y_pred, y_true)));
    return loss;
}