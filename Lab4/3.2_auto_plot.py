import numpy as np
import matplotlib.pyplot as plt


a = np.load('test_3_layers.npy').item()

plt.plot(a['loss'])
plt.show()





