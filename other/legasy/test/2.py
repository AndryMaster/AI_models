import matplotlib.pyplot as plt
import numpy as np

s, e = 2.9, 4.0
R = s
last_x = 0.65
length = 250_000
arr = np.zeros(shape=[length], dtype=np.float64)
x = np.zeros(shape=[length], dtype=np.float64)
arr[0] = last_x
x[0] = R

for i in range(1, length):
    arr[i] = R * arr[i-1] * (1 - arr[i-1])
    R += (e - s) / length
    x[i] = R

plt.xlabel('R scale')
plt.ylabel('Value')
plt.plot(x, arr, 'r,', ms=1, aa=True, alpha=1)
plt.savefig('image1.png', dpi=230, bbox_inches='tight')
plt.show()

# Wiki
# n = 0.01
# y = []
# x = []
# r = 0.01
#
# for j in range(200):
#     for i in range(300):
#         n = 1.0 - r * n * n
#         y.append(n)
#         x.append(r)
#     r += 0.01

# plt.plot(x, y, 'r.', ms=1, aa=True, alpha=0.85)
# plt.show()

# arr = []
# for i in range(30):
#     arr.append(round((y[-i] + 1) / 2, 6))
# print(arr)
