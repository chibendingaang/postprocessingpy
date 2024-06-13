import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./matplotlibrc')
# import matplotlib_inline.backend_inline
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg','pdf');


x = np.arange(0, 40, 0.01)
y = np.cos(2*x) - np.sin(3*x)

plt.figure()
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('samplefig.pdf')
plt.show()

