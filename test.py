import numpy as np
import matplotlib.pyplot as plt

N = 20
# Slope
a = 0.7

dataX = []
X1 = []
X2 = []
for i in range(N):
    x1 = i/N
    X1.append(x1)

dataX.append(X1)

for i, x in enumerate(X1):
    error = np.random.uniform(-0.3,0.3)
    # print('error: ', error)
    x2 = a*x + error
    X2.append(x2)

dataX.append(X2)
dataX = np.transpose(np.array(dataX))
print(dataX)

# plt.plot(X1,X2, 'ro')
# plt.axis([0, 1, 0, 1])
# plt.show()
