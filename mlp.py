## MULTILAYER PERCEPTRON IMPLEMENTATION V 1.0
## HARD CODED VERSION, WILL MAKE IT DYNAMIC IN THE NEXT CELL
from linear_algebra import mat_mult, init_matrix
import math


def sigmoid(x):
  return 1/(1+math.exp(-x))

class mlp:
  # initial values same as the example above
  # cuz im too retarded to figure out how to make this dynamic (again)
  epochs = 10
  eta = 1.0
  inputs = [
      [0.35],
      [0.9]
  ]
  weights = [     [
                   [0.1, 0.8],
                   [0.4, 0.6]
                  ],
                  
                  [
                    [0.3, 0.9]
                  ]
            ]
  bias = 0.0

  def __init__(self):
    pass

  def sigmoid(self, mat): # takes a vector as an input parameter and returns the
                          # sigmoid probability output vector
    mat_output=init_matrix(len(mat), len(mat[0]))
    for i in range(len(mat)):
      for j in range(len(mat[0])):
        mat_output[i][j] = sigmoid(mat[i][j])
    return mat_output



  def forward(self):

    # input layer to hidden layer
    w1 = self.weights[0]
    result1 = self.sigmoid(mat_mult(w1, self.inputs))

    # hidden layer to output layer
    w2 = self.weights[1]
    result2 = self.sigmoid(mat_mult(w2, result1))

    return result2

  


if __name__=='__main__':
  mlp = mlp()

  print(mlp.forward())
