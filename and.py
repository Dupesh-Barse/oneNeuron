


AND = {"x1":[0,0,1,1],
       "x2":[0,1,0,1],
       "y":[0,0,0,1] }

df_AND = pd.DataFrame(AND);df_AND

x,y = prepare_data(df_AND)
ETA = 0.3    #Learning rate 0 to 1
EPOCHS = 10

model_AND = Perceptron(eta=ETA,epochs=EPOCHS)
model_AND.fit(x,y)
_ = model_AND.total_loss()