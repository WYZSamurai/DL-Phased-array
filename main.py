import data1


theta_max = 180
theta_min = 0
scale = 2
batch_size = 3
Fdb, theta = data1.generate(batch_size, scale, theta_max, theta_min)
delta = theta.shape[0]
# print(delta)
# print(Fdb)

for i in range(Fdb.shape[0]):
    data1.plot(theta, Fdb[i], theta_max, theta_min)
