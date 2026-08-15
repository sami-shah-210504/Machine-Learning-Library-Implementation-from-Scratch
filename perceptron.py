## PERCEPTRON IMPLEMENTATION V 3.0
## THE CORRECT VERSION WITH ADDITIONAL FEATURES

import random
def gen_rand(low, up):
  return (low+(up-low)*random.random())


# Using classes (similar to PyTorch) --> dynamic version + additional functions and added printed statements
class perceptron:
  # removed the default attribute initialisation
  # kept it as default parameters

  # also removed 'inputs' list-of-lists attribute since
  # an input is not exactly an attribute of a perceptron

  def __init__(self, learning_rate=0.2,bias = 0.1, num_iters=10):
    # initialise all
    self.eta = learning_rate
    self.epochs = num_iters
    self.bias = bias
    self.weights = list() # weights are initialy an empty list

  def __repr__(self): # copied this dunder method implementation
                      # from google gemini to make model printing neater
        return f"perceptron(max_epochs={self.epochs}, learning_rate={self.eta}, bias={self.bias}, weights={self.weights})"


  def forward(self, x): # takes a list x of feature values as an input i.e x = [x1, x2, x3,... xn]. Returns activation result.
    # net() --> activation() --> prediction

    # net() [weighted sum]
    net = self.bias
    for i in range(len(x)):
      net=net+(x[i]*self.weights[i])

    # activation() [step function]
    if net>=0:
      return 1
    else:
      return 0

  # accuracy method for model evaluation
  def accuracy(self, samples, y_act): # takes a list of samples 'samples' and list of
                                      # target model outputs 'y_act' as an input. Returns accuracy
    correct = 0
    for i in range(len(samples)):
      y_pred = self.predict(samples[i])
      if y_pred == y_act[i]: # if THIS prediction 'y_pred' is correct
        correct+=1
    return (correct/len(y_act)) # accuracy = correct_predictions/total_predictions





# len(x) => number of training samples
# len(x[0]) => number of features per sample
  def train(self, x, y): # x in this case is a list of samples (i.e a list-of-lists)
    # self.inputs = x #inputs stored for now, personal design choice

    # initialise weights to some random number between -1 and 1
    self.weights = list() # weights list is dynamically redeclared and reinitialized
                          # every time train() is called since inputs are now given dynamically
                          # and num_weights = num_input_features
    for i in range(len(x[0])):
      self.weights.append(gen_rand(-1, 1))

    # iterate over epochs = 10
    for i in range(self.epochs):
      acc = self.accuracy(x, y)
      print('Epoch:', i)
      print('Accuracy:',acc,'\n' )
      if acc ==1.0: # observed and actual predictions are all the same now,
        break       # so weights no longer update and no more iterations required



      # iterate over samples
      for j in range(len(x)):

        # # new sample, new deltas
        # delta = list() ----> we are no longer storing deltas in a list.
        # it is now a temp var since there is no need to remember delta values in the long term

        # forward()
        y_pred = self.forward(x[j])

        # error()
        error = y[j] - y_pred

        if error!=0: # y_predicted != y_actual
          # iterate over all feature weights
          for k in range(len(x[0])):

            # delta_weights()
            delta=self.eta*error*x[j][k]
            self.weights[k] = self.weights[k] + delta

        # the bias was originally being updated each time a weight[k] was updated in V 1.0 (i.e bias is updated k times)
        # in the actual perceptron algorithm, bias is updated only once alongside every other weight
        # to correct this, we update bias outside of the if block so it only updates once
        # (if error = 0, delta_bias = 0, so bias value does not change)
        # bias update
        delta_bias = self.eta*error # error corrected -> dont multiply bias in delta_bias
        self.bias = self.bias + delta_bias



  def predict(self, x):
    return self.forward(x)



if __name__=='__main__':
  model = perceptron()
  model.train([
  [1, 0, 1],
  [0, 1, 1],
  [1, 1, 0],
  [0, 0, 1]
  ],
  [1, 0, 1, 0]
  )

  # model.print_preds()

  print("Input [1, 0, 1] --> Output",model.predict([1, 0, 1]))
  print("Input [0, 1, 1] --> Output",model.predict([0, 1, 1]))
  print("Input [1, 1, 0] --> Output",model.predict([1, 1, 0]))
  print("Input [0, 0, 1] --> Output",model.predict([0, 0, 1]))

  print('\n\n')

  model_2 = perceptron()

  x = [[0, 0],
      [0, 1],
      [1, 0],
      [1, 1]]
  y = [0, 1, 1, 1]

  model_2.train(x, y)

  print("OR Gate Approximation")
  print("[0, 0]: ", model_2.predict([0, 0]))
  print("[0, 1]: ", model_2.predict([0, 1]))
  print("[1, 0]: ", model_2.predict([1, 0]))
  print("[1, 1]: ", model_2.predict([1, 1]))

  print('\n\n')

  model_3 = perceptron()
  x = [[0, 0],
      [0, 1],
      [1, 0],
      [1, 1]]
  y = [0, 0, 0, 1]

  model_3.train(x, y)

  print("AND Gate Approximation")
  print("[0, 0]: ", model_3.predict([0, 0]))
  print("[0, 1]: ", model_3.predict([0, 1]))
  print("[1, 0]: ", model_3.predict([1, 0]))
  print("[1, 1]: ", model_3.predict([1, 1]))

  # possible future updates to this code would be to make it more modular
  # i.e make separate functions for the error, weighted_sum and weight_updation
  # calculations and possibly alter the functionality of the attribute
  # 'weights' to not be dynamically reinitialized and redeclared every time
  # train() is called and would instead be one set of static values as soon as
  # an instance of perceptron() is constructed
