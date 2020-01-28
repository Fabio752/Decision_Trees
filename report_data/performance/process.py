import numpy as np

# data in s
# Intel Core i7-8700
train_full = [
    0.745, 0.764, 0.732, 0.745, 0.736, 0.732, 0.740, 0.747, 0.738, 0.735,
    0.737, 0.744, 0.738, 0.730, 0.743, 0.740, 0.743, 0.734, 0.739, 0.735,
    0.731, 0.744, 0.748, 0.741, 0.729, 0.740, 0.744, 0.735, 0.733, 0.735
] 

train_sub = [
    0.113, 0.115, 0.112, 0.118, 0.116, 0.111, 0.117, 0.113, 0.113, 0.116,
    0.115, 0.112, 0.120, 0.115, 0.113, 0.117, 0.114, 0.112, 0.116, 0.118,
    0.115, 0.118, 0.112, 0.115, 0.116, 0.115, 0.112, 0.118, 0.112, 0.118
    
]

train_noisy = [
    0.813, 0.833, 0.818, 0.818, 0.822, 0.837, 0.828, 0.820, 0.819, 0.821,
    0.820, 0.816, 0.816, 0.830, 0.823, 0.820, 0.824, 0.823, 0.819, 0.819,
    0.828, 0.822, 0.826, 0.823, 0.822, 0.827, 0.825, 0.815, 0.821, 0.821
]

fullMean = np.average(train_full)
fullStdDev = np.std(train_full)

subMean = np.average(train_sub)
subStdDev = np.std(train_sub)

noisyMean = np.average(train_noisy)
noisyStdDev = np.std(train_noisy)

print("Data from corona50, 30 samples each")
print("Intel Core i7-8700; Ubuntu 18.04.3 LTS")

print("\n===== train_full =====")
print("{:.3f} +- {:.3f} seconds".format(fullMean, fullStdDev))

print("\n===== train_sub =====")
print("{:.3f} +- {:.3f} seconds".format(subMean, subStdDev))

print("\n===== train_noisy =====")
print("{:.3f} +- {:.3f} seconds".format(noisyMean, noisyStdDev))
