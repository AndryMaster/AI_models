from autoencoder import AutoEncoder
from main import open_csv
import numpy as np

data = open_csv("../datasets/iris/iris.data.csv", delimiter=',')
data = np.array([np.array(list(map(float, param[:-1]))) for param in data])
print(data, data.shape)

hidden_dim = 1
input_dim = len(data[0])
ae = AutoEncoder(input_dim, hidden_dim, epoch=1000)
ae.train(data, batch_size=30)
ae.test([8, 4, 6, 2])
ae.test([6.5, 2.8, 5.4, 1.95])
ae.test([2, 6, 4, 8])
ae.test([1, 1, 1, 1])
