import numpy as np

                                                                                
mu = 1
sigma = 0.1 	
x = np.random.normal(mu, sigma, (500,4))
enumerations = np.arange(0, 500, dtype = int).reshape(500,1)
y = enumerations + np.random.normal(mu, sigma, (500,1))

print(x.shape)
print(y.shape)

final = np.zeros((500,5))
for i in range(5):
    sub_y = np.random.normal(i, 0.1, (100, 1))
    sub_x = np.random.normal(i, 0.1, (100, 4))

    sub_group = np.hstack((sub_x, sub_y))
    final[i: i+100, :] = sub_group

x = final[:, 0:4]
y = final[:, -1]	 

print(x.shape, y.shape)