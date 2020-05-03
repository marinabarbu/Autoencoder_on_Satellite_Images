import pickle
from scipy.spatial import distance
from math import sqrt

with open("F1", 'rb') as f:
    F1 = pickle.load(f) # lista de 2312 vectori de dim 512

with open("F2", 'rb') as f:
    F2 = pickle.load(f)

# print(len(F1))
# print(type(F1))
# print()
#
#
# print(len(F2))
# print(type(F2))
# print(F2[34])

dist_euclid_list = []

for i in range(len(F1)):
    dist = 0.0
    for j in range(len(F1[i])):
        dist += (F1[i][j] - F2[i][j])**2
    dist_euclid_list.append(sqrt(dist))

print(dist_euclid_list)
print(len(dist_euclid_list))