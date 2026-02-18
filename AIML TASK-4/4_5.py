import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 100)

sigmoid = 1 / (1 + np.exp(-z))

plt.plot(z, sigmoid)

plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Probability")

plt.show()
