__author__ = 'Meng'


import sys
print(sys.platform)
print(2**100)


import matplotlib.pyplot as plt
from collections import Counter
c = Counter([6, 4, 0, 0, 0, 0, 0, 1, 3, 1, 0, 3, 3, 0, 0, 0, 0, 1, 1, 0, 0, 0, 3, 2, 3, 3, 2, 5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 2, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 3, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2, 2, 3, 2, 1, 0, 0, 0, 1, 2])
sorted(c.items())
[(0, 50), (1, 30), (2, 9), (3, 8), (4, 1), (5, 1), (6, 1)]
plt.plot(*zip(*sorted(c.items())))
plt.show()



import numpy as np
import math

def drop_zeros(a_list):
    return [i for i in a_list if i>0]

def log_binning(counter_dict,bin_count=35):

    max_x = math.log10(max(counter_dict.keys()))
    max_y = math.log10(max(counter_dict.values()))
    max_base = max([max_x,max_y])

    min_x = math.log10(min(drop_zeros(counter_dict.keys())))

    bins = np.logspace(min_x,max_base,num=bin_count)

    # Based off of: http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    bin_means_y = (np.histogram(counter_dict.keys(),bins,weights=counter_dict.values())[0] / np.histogram(counter_dict.keys(),bins)[0])
    bin_means_x = (np.histogram(counter_dict.keys(),bins,weights=counter_dict.keys())[0] / np.histogram(counter_dict.keys(),bins)[0])

    return bin_means_x,bin_means_y


import networkx as nx
ba_g = nx.barabasi_albert_graph(10000,2)
ba_c = nx.degree_centrality(ba_g)
# To convert normalized degrees to raw degrees
# ba_c = {k:int(v*(len(ba_g)-1)) for k,v in ba_c.iteritems()}
ba_c2 = dict(Counter(ba_c.values()))

ba_x,ba_y = log_binning(ba_c2,35)

plt.xscale('log')
plt.yscale('log')
plt.scatter(ba_x,ba_y,c='r',marker='s',s=50)
plt.scatter(ba_c2.keys(),ba_c2.values(),c='b',marker='x')
plt.xlim((1e-4,1e-1))
plt.ylim((.9,1e4))
plt.xlabel('Connections (normalized)')
plt.ylabel('Frequency')
plt.show()



from scipy.stats import powerlaw
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
a = 1.66
mean, var, skew, kurt = powerlaw.stats(a, moments='mvsk')
x = np.linspace(powerlaw.ppf(0.01, a), powerlaw.ppf(0.99, a), 100)
ax.plot(x, powerlaw.pdf(x, a),'r-', lw=5, alpha=0.6, label='powerlaw pdf')


# Dec 15, 2015 Ch4

import math
math.pi
import random
random.choice(['a','b','c','d'])
S = 'Spam'
S[:-1]

B = bytearray(b'spam')
B.extend(b'eggs')
B.decode()

line = 'aaa,bbb,ccc,dd\n'
line.rstrip().split(',')
line

'%s,eggs, and %s'%('spam', 'SPAM!')

'{0}, eggs, and {1}'.format('spam','SPAM')

'{:,.2f}'.format(296999.2567)
'%.2f | %+05d' % (3014159,+42)

dir(S)

ord('A')
ord('a')
ord('H')
S = 'A\nB\nC\tD'
print(S)
S = 'A\0B\0C\tD'
S

#nomal str strings are Unicode text
'sp\xc4m'

print('a\xc4c')
print(b'a\xc4c')
b'a\x01c'
u'sp\u00c4m'

'spam'.encode('utf8')
'spam'.encode('utf16')
'sp\xc4\u00c4\U000000c4m'

import re
match = re.match('Hello[\t]*(.*)world','Hello   Python world')
match.group(1)
match.group(0)
match.groups()

D = {'a':1, 'b':2,'c':3}
Ks = list(D.keys())
Ks
Ks.sort()
for key in Ks:
    print(key,'=>', D[key])

for c in 'spam':
    print(c.upper())

x=4
while x>0:
    print('spam!'*x)
    x-=1

T = (1,2,3,4)
T = T +(5,6)

{n ** 2 for n in [1,2,3,4,5]}

import decimal
d = decimal.Decimal('3.141')
d+1

from fractions import Fraction
f = Fraction(1,3)
f+1

class Worker:
    def __init__(self, name, pay):
        self.name = name
        self.pay = pay
    def lastName(self):
        return self.name.split()[-1]
    def giveRaise(self, percent):
        self.pay*= (1.0+ percent)

bob = Worker("Bob Smith", 5000)
sue = Worker("Sue Jones", 6000)

set(bob.name)-set(bob.lastName())
sue.giveRaise(.10)
sue.pay

1//2
10.1//4

import math
math.floor(-2.5)

x = 1
x<<10
math.pi

import random
random.choice(['a','b','c','d','e','f'])
suits = ['a','b','c','d','e','f']
random.shuffle(suits)
suits