import numpy as np
from matplotlib import pyplot as plt

l = np.random.rand()
x = 1 - l
y = np.random.rand()*x
if l > 1 or y > 1-l or l < 0 or y < 0:
    raise ValueError("Invalid input: l must be < 1, y must be < 1-l, and both >= 0.")

z = 1 - l - y
X = np.array([[l], [y], [z]])
print("Initial probabilities of weather conditions (Sunny, Cloudy, Rainy): ")
print(X)
A = np.array([[0.5, 0.6, 0.1], [0.25, 0.25, 0.1], [0.25, 0.15, 0.8]])
while True:
    j = int(input("Enter how many interations you want: "))
    if j > 0:
        break
    else:
        print("Please enter a positive integer.")
for i in range(j):
    print(i+1)
    Xn = A @ X
    print(Xn)
    X = Xn
    
