import matplotlib.pyplot as plt
import numpy as np

x = np.array([i for i in range(1, 13)])

y = np.array([168.789, 89.47, 45.4895, 35.478, 30.231, 30.231, 30.231, 30.231, 30.231, 30.231, 30.231, 30.231])
z = np.array([yi - 10 for yi in y])
v = np.array([yi - 20 for yi in y])

plt.plot(x, y, color='r', label='sin', marker='s')
plt.plot(x, z, color='g', label='cos', marker='o')
plt.plot(x, v, color='y', label='tang')

plt.xlabel("Tread")
plt.ylabel("Time")
plt.title("Sine and Cosine functions")

plt.legend()

plt.show()