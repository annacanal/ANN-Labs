import numpy as np
import matplotlib.pyplot as plt


a = np.load('test_3_layers_75epochs_0.3lr.npy').item()

plt.plot(a['loss'])
plt.show()





