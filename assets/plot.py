import matplotlib.pyplot as plt

num_sequences = [10,100,1000,10000]
train_avg = [-122.54156723373207,-110.31599954262039,-101.07201479899955,-95.43736971788366]
valid_avg = [-131.64934332803858,-116.09167374707617,-105.3081927138837,-98.52825144371127]

plt.plot(num_sequences,train_avg,label="Train Average Log Likelihood")
plt.plot(num_sequences,valid_avg,label="Validation Average Log Likelihood")
plt.xlabel("Num of Sequences")
plt.ylabel("Average Log Likelihood")
plt.legend()
plt.show()