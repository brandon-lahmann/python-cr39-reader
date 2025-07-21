from reader import ScanData
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

data = ScanData('O109043_CPS2_d9w_dw7813_4.5hr_40x_s0.cpsa', c_bounds=(0, 50), x_bounds=(0.0, 1.2), y_bounds=(-1.5, 1.0))
tracks = data.tracks

plt.hist2d(tracks['d'], tracks['c'], bins=[500, 50], norm=LogNorm())
plt.xlim([0, 20])
plt.show()

tracks = tracks[tracks['c'] < 20]
plt.hist2d(tracks['x'], tracks['y'], bins=100)
plt.show()

print(data.header)
print(data.trailer)
