#CREATING PEREPTRON CLASS FOR  OPERATION:

class Perceptron:
  def __init__(self , eta , epochs):
    """Intializing Perceptron"""
    self.weights = np.random.randn(3) * 1e-4                          #intializing random weight by using random normal samples and 1e-4 (10^-4)
    print(f"Intial Weight before training : {self.weights} ")         # use f to use { } instead of .format()
    self.eta = eta                                                    # learning rate
    self.epochs = epochs                                              # number of iteration / epochs 

  def activationFunction(self,inputs,weights):
    z = np.dot(inputs , weights)                                      # z= W * X  (matrix dot product)
    return np.where(z > 0 , 1,0)

  def fit(self, x , y ):
    self.x = x                                                         #self use because it can be used in entire class, not only for fit function
    self.y = y

    x_with_bias = np.c_[self.x, -np.ones((len(self.x),1))]             #c_[]: Concatenate , x_with_bias = -1
    print(f"x with bias : {x_with_bias}")

    for epoch in range(self.epochs):
      print("--"*10)
      print(f"epoch: {epoch}")
      print("--"*10)

      y_hat = self.activationFunction(x_with_bias , self.weights)     # FWD propagation
      print(f"Predicted Value after Forward pass: {y_hat}")
      
      self.error = self.y - y_hat 
      print(f"error: {self.error}")
      #Now new weights : Backward Propagation 
      self.weights = self.weights + self.eta * np.dot(x_with_bias.T,self.error)   #.T : transpose of matrix
      print(f"updated weights after epoch: {epoch}/{self.epochs} : {self.weights}")
      print("########"*10)  
  
  def predict(self,X):
    x_with_bias = np.c_[X,-np.ones((len(X),1))]
    return self.activationFunction(x_with_bias , self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"total loss : {total_loss} ")
    return total_loss
              
