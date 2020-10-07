from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from plotting import LatexifyMatplotlib as lm

dd_31 = genfromtxt('DD_31', delimiter=',')
dd_32 = genfromtxt('DD_32', delimiter=',')

s11_31 = 20 * np.log10(np.abs(dd_31[:, 0] + 1j * dd_31[:, 1]))
s11_32 = 20 * np.log10(np.abs(dd_32[:, 0] + 1j * dd_32[:, 1]))

x1 = [865, 869.5, 869.5, 865, 865]
y1 = [-50, -50, -10, -10, -50]

y = np.ones(shape=401) * -10
x = np.linspace(800, 900, num=401)

idx_31 = np.argwhere(np.diff(np.sign(y - s11_31))).flatten()
idx_32 = np.argwhere(np.diff(np.sign(y - s11_32))).flatten()

plt.plot(x, s11_31, label="Antenna 1")
plt.plot(x, s11_32, label="Antenna 2")
plt.plot(x, y)
plt.plot(x[idx_31], s11_31[idx_31], 'bx')
plt.plot(x[idx_32], s11_32[idx_32], 'bx')
plt.grid(True)
plt.annotate(x[idx_31][0], xy =(845, -13))
plt.annotate(x[idx_31][1], xy =(875, -13))
# plt.fill(x1,y1)
plt.legend()
lm.save("s11.tex", scale_legend=0.7, show=True, plt=plt)
