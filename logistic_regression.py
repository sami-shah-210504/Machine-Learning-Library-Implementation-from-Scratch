# logistic regression -> binary classification
# based on an input X, the output is a predicted probability that X
# belongs to class 1 or 0, A or B etc.
# unlike the perceptron which simply directly classifying an input,
# log reg also provides a confidence level that an input is classified

## LOGISTIC REGRESSION IMPLEMENTATION V 2.0
## COMPLETE AND MATHEMATICALLY CORRECT VERSION
## FUTURE VERSIONS MAY EMPLOY MINI-BATCHES, NUMERICAL STABILITY, L2 REGULARIZATION, MOMENTUM, ADAM, VALIDATION SETS, ETC
## AND STORE LOSSES IN A LIST

import random
def gen_rand(low, up):
  return (low+(up-low)*random.random())

import math
def sigmoid(x):
  return 1/(1+math.exp(-x))

def sigmoid_activation(x):
  if x>=0.5:
    return 1
  else:
    return 0

class logistic_regression:
  # constructor and __repr__ for this class is identical to the perceptron
  def __init__(self, learning_rate=0.2,bias = 0.1, num_iters=10):
    # initialise all
    self.eta = learning_rate
    self.epochs = num_iters
    self.bias = bias
    self.weights = list() # weights are initialy an empty list

  def __repr__(self): # copied this dunder method implementation
                      # from google gemini to make model printing neater
        return f"logistic_regression(max_epochs={self.epochs}, learning_rate={self.eta}, bias={self.bias}, weights={self.weights})"

  # unlike a percetron, the forward method for
  # a log_reg model returns a probability rather than
  # a classification/step function result
  def forward(self, x): # x is a list of input feature values i.e x1, x2, ..., xn
      # net() [weighted sum]
      net = self.bias
      for i in range(len(x)):
        net=net+(x[i]*self.weights[i])

      # calling sigmoid() to calc probability of inputs belonging to class 1
      return sigmoid(net)

  def predict(self, x):
    return sigmoid_activation(self.forward(x)) # predict() method returns a class (1 or 0)
                                               # based on sigmoid_activation() func

  # accuracy method for model evaluation
  def accuracy(self, samples, y_act): # takes a list of samples 'samples' and list of
    correct = 0                       # target model outputs 'y_act' as an input. Returns accuracy
    for i in range(len(samples)):
      y_pred = self.predict(samples[i])
      if y_pred == y_act[i]: # if THIS prediction 'y_pred' is correct
        correct+=1
    return (correct/len(y_act)) # accuracy = correct_predictions/total_predictions

  def train(self, x, y): # x-> list of samples (list-of-lists). y-> list of outputs



    # initialise weights to some random number between -1 and 1
    self.weights = list() # weights list is dynamically redeclared and reinitialized
                          # every time train() is called since inputs are now given dynamically
                          # and num_weights = num_input_features
    for i in range(len(x[0])):
      self.weights.append(gen_rand(-1, 1))


     # iterate over epochs = 10
    for i in range(self.epochs):
      dataset_loss = 0 # initialize training loss to zero
                     # so we can calculate cost at the end of each epoch
                     # (this is a correction. V 1.0 only printed individual loss of the last sample at the end of each epoch)

      # iterate over samples
      for j in range(len(x)):

        # # new sample, new deltas
        # delta = list() ----> we are no longer storing deltas in a list.
        # it is now a temp var since there is no need to remember delta values in the long term

        # forward()
        y_pred = self.forward(x[j])


        # binary cross entropy loss (log loss)
        loss = -((y[j]*math.log(y_pred))+((1-y[j])*math.log(1-y_pred)))
        dataset_loss = dataset_loss+loss


        # error()
        error_signal = y_pred-y[j]

        # compute gradients
        # iterate over all feature weights
        for k in range(len(x[0])):
          gradient = error_signal*x[j][k]
          self.weights[k] = self.weights[k] - (self.eta*gradient) # gradient descent algo
        gradient = error_signal
        self.bias = self.bias - (self.eta*gradient)

      # for log_reg train() method, we also print
      # the binary cross entropy loss which should
      # ideally decrease with each iteration/epoch
      dataset_loss=dataset_loss/len(x) # sum_of_sample_losses/number_of_samples
      acc = self.accuracy(x, y)
      print('Epoch:', i)
      print('Loss', dataset_loss)
      print('Accuracy:',acc,'\n' )



if __name__=='__main__':
#   | Sample | Features | Target |
# | ------ | -------- | ------ |
# | 1      | [2,1]    | 1      |
# | 2      | [1,2]    | 1      |
# | 3      | [-1,-2]  | 0      |
# | 4      | [-2,-1]  | 0      |
  logreg_model = logistic_regression()
  logreg_model.train([
      [2, 1],
      [1, 2],
      [-1, -2],
      [-2, -1]
      ],
    [1, 1, 0, 0]
  )
  print(logreg_model)
  print('\n')

  print('[2, 1]:',logreg_model.predict([2, 1]))
  print('[1, 2]:',logreg_model.predict([1, 2]))
  print('[-1, -2]:',logreg_model.predict([-1, -2]))
  print('[-2, -1]:',logreg_model.predict([-2, -1]))



