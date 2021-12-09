


x,y = prepare_data(df_AND)
ETA = 0.3    #Learning rate 0 to 1
EPOCHS = 10

model_AND = Perceptron(eta=ETA,epochs=EPOCHS)
model_AND.fit(x,y)
_ = model_AND.total_loss()