import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

import scipy.stats as stats;

pr = np.array([0.025, 0.05, 0.10, 0.10, 0.10, 0.2, 0.2, 0.2, 0.025]);
ret = np.array([-0.4, -0.2, -0.1, 0, 0.05, 0.1, 0.15, 0.20, 0.30]);


sample = np.array([]);

# Generate the sample with the given priors
for i in range(pr.size):
    # print(pr[i]);
    for j in range(int(1000*pr[i])):
        sample = np.append(sample, ret[i])

plt.figure();
plt.hist(sample);
plt.show();



print(round(np.mean(sample), 3))
print(round(np.std(sample), 3))
print(round(stats.skew(sample), 3))
print(round(stats.kurtosis(sample), 3))


quants = np.quantile(sample, [0.01, 0.05]);

print(quants)
print(quants[0])
print(quants[1])

VaR = np.array([(-0.4*0.01) / 0.01, (-0.4*0.025 -0.2*0.025)/0.05])
print("var\n")
print(VaR)
# print(np.mean(sample[sample < quants[0]]))

print(quants*100);



def utility(w, gamma = 2):
    return np.power(w, 1 - gamma)/(1 - gamma)

def inverse_utility(u, gamma = 2):
    return np.exp(np.log(u*(1 - gamma))/(1 - gamma))


# Test the inverse utility function
# print(inverse_utility(utility(400)));


rf = {};
W0 = 100;


# 100% either in risky or risk-free

for ra in range(2, 15):
    expected_utility = np.mean([utility((1 + x) * W0, ra) for x in sample]);
    rf[ra] = inverse_utility(expected_utility, ra) / W0 - 1;

print('The breakeven risk-free asset is :' + str(rf[2]));

plt.figure();
print(np.array(rf.values()));

x, y = zip(*sorted(rf.items()));

plt.plot(x, y);
plt.show();
print(rf)

# The utility increases with gamma > 2

# Part (e)



gamma = 2;
# Start investing all the wealth in the risk-less asset at approx 7.2% return
# it is higher than the equivalent risk-free interest rate providing the same
# utility as the expectation of the utility of the risky return.
for rf in np.arange(0.070, 0.074, 0.001):
    utility_pi = {};
    # invest pi in the risk-less asset
    for pi in np.arange(0, 1, 0.01):
        utility_pi[pi] = np.mean([utility((1 + x) * W0 * (1 - pi) +
                              (1 + rf) * W0 * pi, gamma) for x in sample]);
    pi_star = max(utility_pi, key = utility_pi.get);
    print(str(pi_star) + ' ' + str(rf));







