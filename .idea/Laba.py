import math
import numpy as np
import statistics
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from scipy.stats import expon
from scipy.stats import geom


def BernoulliGen(n, q):
    sample = np.random.rand(n)
    for i in range(0, len(sample)):
        if sample[i] > q:
            sample[i] = 1
        else:
            sample[i] = 0
    return sample

def EntropyCalculation(n, sample):
    unique = []
    count = 0
    for i in range(1, n):
        if sample[i] != sample[i - 1]:
            unique.append(sample[i - 1])
            if sample[i] not in unique:
                unique.append(sample[i])
    frequence = [0] * len(unique)

    for i in sample:
        try:
            frequence[unique.index(i)] += 1
        except ValueError:
            continue
    for i in range(0, len(frequence)):
        frequence[i] = frequence[i]/n


    entropy = 0
    for i in frequence:
        if i != 0:
            entropy += i * math.log(i)
    return -1 * entropy

def BernoulliEntropy(p):
    return -(1 - p) * math.log(1 - p) - p * math.log(p)

p = 0.5 - 2/1000

trueEntropy = np.array([BernoulliEntropy(p)] * 100)

variancePython = []
varianceUser = []
print('Bernoulli Distribution \n')
for l in range(10, 1001, 10):
    entropyPython = []
    entropyUser = []
    for _ in range(0, 100):
        pythonBernoulli = bernoulli.rvs(size=l, p=p)
        UserBernoulli = BernoulliGen(l, 1 - p)
        entropyPython.append(EntropyCalculation(l, sorted(pythonBernoulli)))
        entropyUser.append(EntropyCalculation(l, sorted(UserBernoulli)))

    print('Размер выборки: ' + str(l) + ' средняя энтропия для распределения, созданного с использованием встроенной функции ' + str(entropyPython[-1]) +
                                        ', \n средняя энтропия для распределения, созданная с использованием генерации случайной величины ' + str(entropyUser[-1]) )

    variancePython.append(statistics.variance(trueEntropy - entropyPython))
    varianceUser.append(statistics.variance(trueEntropy - entropyUser))
    print('Дисперсия отклонений: ' + str(variancePython[-1]) + ', ' + str(varianceUser[-1]))

plt.figure(figsize=(10, 5))
plt.plot(range(10, 1001, 10), variancePython)
plt.xlabel('Array size')
plt.ylabel('Data by Python')
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(range(10, 1001, 10), varianceUser)
plt.xlabel('Array size')
plt.ylabel('Data by User')
plt.show()

#геометрическое распределение
def GeomGen(n, q):
    arr = np.random.rand(n)
    for i in range(0, len(arr)):
        arr[i] = arr[i]/math.log(q)
    return arr

def GeomEntropy(p):
    return -math.log(p, 2) - ((1 - p)/p) * math.log((1 - p), 2)

true_Entropy = np.array([GeomEntropy(p)] * 100)

variance_Python = []
variance_User = []
print('Geom Distribution \n')
for l in range(10, 1001, 10):
    entropy_Python = []
    entropy_User = []
    for _ in range(0, 100):
        pythonGeom = geom.rvs(size=l, p=p)
        UserGeom = GeomGen(l, 1 - p)
        #print (str(UserGeom))
        entropy_Python.append(EntropyCalculation(l, sorted(pythonGeom)))
        entropy_User.append(EntropyCalculation(l, sorted(UserGeom)))
        #print('\n')

    print('Размер выборки: ' + str(l) + ' средняя энтропия для распределения, созданного с использованием встроенной функции ' + str(entropy_Python[-1]) +
          ', \n средняя энтропия для распределения, созданная с использованием генерации случайной величины ' + str(entropy_User[-1]) )

    variance_Python.append(statistics.variance(true_Entropy - entropy_Python))
    variance_User.append(statistics.variance(true_Entropy - entropy_Python))
    print('Дисперсия отклонений: ' + str(variance_Python[-1]) + ', ' + str(variance_Python[-1]))


plt.figure(figsize=(10, 5))
plt.plot(range(10, 1001, 10), variance_Python)
plt.xlabel('Array size')
plt.ylabel('Array by Python')
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(range(10, 1001, 10), variance_Python)
plt.xlabel('Array size')
plt.ylabel('Array by User')
plt.show()


















