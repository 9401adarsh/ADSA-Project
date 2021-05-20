import matplotlib.pyplot as plt
import sys
import numpy as np

file_name1 = "Insert-WH"
file_name2 = "Insert-BH"
file_name3 = "Insert-SH"

filelines1 = open(f"./{file_name1}.txt").readlines()
filelines2 = open(f"./{file_name2}.txt").readlines()
filelines3 = open(f"./{file_name3}.txt").readlines()

random_times1 = [filelines1[i] for i in range(0, len(filelines1))]
random_times2 = [filelines2[i] for i in range(0, len(filelines2))]
random_times3 = [filelines3[i] for i in range(0, len(filelines3))]

sizes = [int(i.split(':')[2]) for i in random_times1]
random_times1 = [float(i.split(':')[-1]) for i in random_times1]
random_times2 = [float(i.split(':')[-1]) for i in random_times2]
random_times3 = [float(i.split(':')[-1]) for i in random_times3]


plt.plot(sizes, random_times1,color = "red")
plt.plot(sizes, random_times2,color = "blue")
plt.plot(sizes, random_times3,color = "green")

#plt.plot(sizes, np.log(sizes), color="black")


plt.title('Insert times comparison')
plt.legend(['Weak Heap', 'Binomial Heap', 'Soft Heap'])

plt.xlabel("no. of nodes")
plt.ylabel("Time")
plt.show()
