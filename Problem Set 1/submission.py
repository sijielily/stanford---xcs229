import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
  """Base class for linear models.

  Attributes:
    theta: np.ndarray, dtype=np.float64, shape=(n_features,). Weights vector for
      the model.
  """

  def __init__(self, theta=None):
    """
    Args:
      theta: (See class definition)
    """
    self.theta = theta

  def fit(self, x, y):
    """Fits the linear model to x -> y using np.linalg.solve.

    Remember to update self.theta with the fitted model parameters.

    Args:
      x: np.ndarray, dtype=np.float64, shape=(n_examples, n_features). Inputs.
      y: np.ndarray, dtype=np.float64, shape=(n_examples,). Outputs.

    Returns: Nothing
    """
    pass
    # *** START CODE HERE ***
    x_transpose = np.transpose(x)
    theta = np.linalg.solve(np.dot(x_transpose,x),np.dot(x_transpose,y))
    self.theta = theta
    # *** END CODE HERE ***

  def predict(self, x):
    """ Makes a prediction given a new set of input features.

    Args:
      x: np.ndarray, dtype=np.float64, shape=(n_examples, n_features). Model input.

    Returns: np.ndarray, dtype=np.float64, shape=(n_examples,). Model output.
    """
    pass
    # *** START CODE HERE ***
    return(np.dot(x,self.theta))
    # *** END CODE HERE ***

  @staticmethod
  def create_poly(k, x):
    """ Generates polynomial features of the input data x.

    Args:
      x: np.ndarray, dtype=np.float64, shape=(n_examples,). Training inputs.

    Returns: np.ndarray, dtype=np.float64, shape=(n_examples, k+1). Polynomial
      features of x with powers 0 to k (inclusive).
    """
    pass
    # *** START CODE HERE ***
    
    #check dimension of inputs to make sure it has shape=(n_examples,)
    if x.ndim > 1:
        x = x[:,0]
    
    new_list = []
    for i in range(k + 1):
        new_list.append(x**i)
    new_array = np.array(new_list)

    return np.transpose(new_array)
    # *** END CODE HERE ***

  @staticmethod
  def create_sin(k, x):
    """ Generates sine and polynomial features of the input data x.

    Args:
      x: np.ndarray, dtype=np.float64, shape=(n_examples,). Training inputs.

    Returns: np.ndarray, dtype=np.float64, shape=(n_examples, k+2). Sine (column
      0) and polynomial (columns 1 to k+1) features of x with powers 0 to k
      (inclusive).
    """
    pass
    # *** START CODE HERE ***
    
    #check dimension of inputs to make sure it has shape=(n_examples,)
    if x.ndim > 1:
        x = x[:,0]
        
    #create sine for first row
    sine = np.sin(x)
    
    new_list = []
    for i in range(k + 1):
        new_list.append(x**i)
    new_array = np.array(new_list)
    
    #add the first row 
    final_array = np.vstack((sine,new_array))
    return np.transpose(final_array)
    
    # *** END CODE HERE ***

def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
  train_x,train_y=util.load_dataset(train_path,add_intercept=False)
  plot_x = np.ones([1000, 1])
  plot_x[:, 0] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
  plt.figure()
  plt.scatter(train_x, train_y)

  for k in ks:
      '''
      Our objective is to train models and perform predictions on plot_x data
      '''
      # *** START CODE HERE ***
      model = LinearModel()
      
      #create_poly
# =============================================================================
#       train_features = model.create_poly(k, train_x)
#       model.fit(train_features,train_y)        
#       plot_features = model.create_poly(k, plot_x)
#       plot_y = model.predict(plot_features)
# =============================================================================

      
      #create_sin
      train_features = model.create_sin(k, train_x)
      model.fit(train_features,train_y) 
      plot_features = model.create_sin(k, plot_x)
      plot_y = model.predict(plot_features)
      # *** END CODE HERE ***
      '''
      Here plot_y are the predictions of the linear model on the plot_x data
      '''
      plt.ylim(-2, 2)
      plt.plot(plot_x[:, 0], plot_y, label='k=%d' % k)

  plt.legend()
  plt.savefig(filename)
  plt.clf()


def main(train_path, small_path, eval_path):
  '''
  Run all experiments
  '''
  run_exp(train_path, False, [3], 'large-poly3.png')
  run_exp(train_path, True, [1, 2, 3, 5, 10, 20], 'large-sine.png')
  run_exp(train_path, False, [1, 2, 3, 5, 10, 20], 'large-poly.png')
  run_exp(small_path, True, [1, 2, 3, 5, 10, 20], 'small-sine.png')
  run_exp(small_path, False, [1, 2, 3, 5, 10, 20], 'small-poly.png')

if __name__ == '__main__':
  main(train_path='train.csv',
      small_path='small.csv',
      eval_path='test.csv')


