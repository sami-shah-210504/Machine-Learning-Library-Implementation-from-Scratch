## MULTILAYER PERCEPTRON IMPLEMENTATION V 2.0
## DYNAMIC VERSION
import math
from linear_algebra import *

def sigmoid(x):
  return 1/(1+math.exp(-x))

class mlp:

  def __init__(self, epochs=10, eta=1.0, inputs=[[0.35],[0.9]], weights=[[[0.1, 0.8],[0.4, 0.6]],[[0.3, 0.9]]], bias = [[[0.0],[0.0]], [[0.0]]]):
    self.epochs = epochs
    self.eta = eta
    self.inputs = inputs
    self.weights = weights
    self.bias = bias
    self.num_layers = len(self.weights)
    self.z = []
    self.a = []
    for i in range(self.num_layers):
      self.z.append(None)
      self.a.append(None)

  def sigmoid(self, mat):
    mat_output=init_matrix(len(mat), len(mat[0]))
    for i in range(len(mat)):
      for j in range(len(mat[0])):
        mat_output[i][j] = sigmoid(mat[i][j])
    return mat_output

  def sigmoid_derivative(self, mat):
    mat_output=init_matrix(len(mat), len(mat[0]))
    for i in range(len(mat)):
      for j in range(len(mat[0])):
        sig = sigmoid(mat[i][j])
        mat_output[i][j] = sig*(1-sig)
    return mat_output

  def forward(self, x=None):
    if x is None:
      x = self.inputs
    self.inputs = x

    prev_activation = self.inputs
    for i in range(self.num_layers):
      w = self.weights[i]
      b = self.bias[i]
      self.z[i] = mat_add(mat_mult(w, prev_activation), b)
      self.a[i] = self.sigmoid(self.z[i])
      prev_activation = self.a[i]

    return self.a[self.num_layers-1]

  def loss(self, y_act, y_pred):
    return -((y_act*math.log(y_pred))+(1-y_act)*math.log(1-y_pred))

  def output_error(self, y_pred, y_act):
    return y_pred - y_act

  def output_gradients(self, y_act, y_pred):
    error = self.output_error(y_pred, y_act)
    delta = [[error]]

    prev_activation = self.a[self.num_layers-2]
    prev_activation_transpose = transpose(prev_activation)
    dW = mat_mult(delta, prev_activation_transpose)
    db = delta

    return dW, db, delta

  def hidden_gradients(self, delta_next, layer_index):
    w_next = self.weights[layer_index+1]
    w_next_transpose = transpose(w_next)
    propagated_error = mat_mult(w_next_transpose, delta_next)
    sig_deriv_z = self.sigmoid_derivative(self.z[layer_index])
    delta = hadamard_prod(propagated_error, sig_deriv_z)

    if layer_index == 0:
      prev_activation = self.inputs
    else:
      prev_activation = self.a[layer_index-1]

    prev_activation_transpose = transpose(prev_activation)
    dW = mat_mult(delta, prev_activation_transpose)
    db = delta

    return dW, db, delta

  def update_params(self, param, gradient):
    scaled_gradient = scale_matrix(gradient, self.eta)
    return mat_sub(param, scaled_gradient)

  def train_step(self, x, y_act):
    y_pred_matrix = self.forward(x)
    y_pred = y_pred_matrix[0][0]

    current_loss = self.loss(y_act, y_pred)

    dW_list = []
    db_list = []
    for i in range(self.num_layers):
      dW_list.append(None)
      db_list.append(None)

    output_layer_index = self.num_layers-1
    dW, db, delta = self.output_gradients(y_act, y_pred)
    dW_list[output_layer_index] = dW
    db_list[output_layer_index] = db

    for i in range(output_layer_index-1, -1, -1):
      dW, db, delta = self.hidden_gradients(delta, i)
      dW_list[i] = dW
      db_list[i] = db

    for i in range(self.num_layers):
      self.weights[i] = self.update_params(self.weights[i], dW_list[i])
      self.bias[i] = self.update_params(self.bias[i], db_list[i])

    return current_loss

  def train(self, x, y_act):
    losses = []
    for epoch in range(self.epochs):
      current_loss = self.train_step(x, y_act)
      losses.append(current_loss)
    return losses

if __name__ == "__main__":
  x = [
        [[0.0],[0.0]],
        [[0.0],[1.0]],
        [[1.0],[0.0]],
        [[1.0],[1.0]]
    ]
  y = [0, 1, 1, 0]

  mlp = mlp(epochs=10000, eta=0.5)
  print("Initial weights:", mlp.weights)
  print("Initial bias:", mlp.bias)

  losses = []
  for epoch in range(mlp.epochs):
    epoch_loss = 0
  for i in range(len(x)):
    current_loss = mlp.train_step(x[i], y[i])
    epoch_loss = epoch_loss + current_loss
    epoch_loss = epoch_loss / len(x)
    losses.append(epoch_loss)

  print("Final losses:", losses)

  for i in range(len(x)):
    prediction = mlp.forward(x[i])
    print("Input:", x[i], "Predicted:", prediction[0][0], "Actual:", y[i])  
