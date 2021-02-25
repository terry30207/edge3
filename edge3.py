import numpy
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from pathlib import Path


numpy.random.seed(7)
(X_train, y_train),(X_test, y_test) = mnist.load_data()
X_train = X_train / 255
X_test = X_test / 255
model = load_model(str(Path(__file__).resolve().parent.parent)+'\\aggr\\global.h5')
dataset2_x = X_train
dataset2_y = y_train
model.fit(dataset2_x, dataset2_y, epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
model.save("edge3.h5")
