__author__ = 'mhan0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
path = 'ch02/usagov_bitly_data2012-03-16-1331923249.txt'
records = [json.loads(line) for line in open(path)]
records[0]
records[0]['tz']

time_zones = [rec['tz'] for rec in records if 'tz' in rec]
len(time_zones)


#This function return a dictionary includes counts and the terms
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] +=1
        else:
            counts[x] = 1
    return counts

get_counts(time_zones)
len(set(time_zones))

from collections import defaultdict
def get_counts2(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts

tz2 = get_counts2(time_zones)
tz2['America/New_York']

def top_counts(count_dict, n = 10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]
top_counts(tz2)

from collections import Counter
counts = Counter(time_zones)
counts.most_common(10)

from pandas import DataFrame, Series
import pandas as pd
frame = DataFrame(records)
frame['tz'][:10]
frame['a'][:10]


clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
tz_counts[:10]

tz_counts[:10].plot(kind='barh', rot = 0)

cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')
operating_system[:5]
by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
agg_counts[:10]

indexer = agg_counts.sum(1).argsort()
indexer[:10]

count_subset = agg_counts.take(indexer)[-10:]
count_subset

count_subset.plot(kind='barh', stacked =True)
normed_subset = count_subset.div(count_subset.sum(1), axis =0)
normed_subset.plot(kind='barh',stacked = True)


# Movie Data Study
import pandas as pd
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('ch02/movielens/users.dat', sep ='::', header = None, names = unames, engine = 'python')

unames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('ch02/movielens/ratings.dat', sep ='::', header = None, names = unames, engine = 'python')

unames = ['movie_id', 'title', 'genres']
movies = pd.read_table('ch02/movielens/movies.dat', sep ='::', header = None, names = unames, engine = 'python')

users[:5]
ratings[:5]
ratings[:5]

data = pd.merge(pd.merge(ratings,users), movies)
data
data.ix[0]

# C:\Users\mhan0\AppData\Local\Continuum\Anaconda3\lib\site-packages\pandas\util\decorators.py:81: FutureWarning: the 'rows' keyword is deprecated, use 'index' instead
#   warnings.warn(msg, FutureWarning)
mean_ratings = data.pivot_table('rating', index = 'title', columns = 'gender', aggfunc = 'mean')
mean_ratings[:10]

rating_by_title = data.groupby('title').size()
active_titles = rating_by_title.index[rating_by_title>= 250]

mean_ratings = mean_ratings.ix[active_titles]

top_female_ratings = mean_ratings.sort_index(by = 'F', ascending=False)
top_female_ratings[:10]

mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']

sorted_by_diff = mean_ratings.sort_index(by = 'diff')

sorted_by_diff[:15]

sorted_by_diff[::-1][:15]


#The third case study:
names1880 = pd.read_csv('ch02/names/yob1880.txt', names = ['name', 'sex', 'births'])
names1880.groupby('sex').births.sum()
years = range(1880,2011)
pieces = []
columns = ['name','sex','births']

#The way to read many data
for year in years:
    path = 'ch02/names/yob%d.txt' % year
    frame = pd.read_csv(path, names = columns)

    frame['year'] = year
    pieces.append(frame)
names = pd.concat(pieces,ignore_index = True)

names

total_births = names.pivot_table('births', index = 'year', columns = 'sex', aggfunc = sum)
total_births.tail()
total_births.plot(title = 'Total births by sex and year')

names

def add_prop(group):
    births = group.births.astype(float)

    group['prop'] = births/births.sum()
    return group
names = names.groupby(['year','sex']).apply(add_prop)
names

def get_top1000(group):
    return group.sort_index(by= 'births', ascending= False)[:1000]
grouped = names.groupby(['year','sex'])
top1000 = grouped.apply(get_top1000)

pieces = []
for year, group in names.groupby(['year','sex']):
    pieces.append(group.sort_index(by='births', ascending=False)[:1000])
top1000 = pd.concat(pieces, ignore_index=True)

boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']

total_births = top1000.pivot_table('births', index = 'year', columns = 'name', aggfunc = sum)

subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]

subset.plot(subplots = True, figsize = (12, 10), grid = False, title= "Number of births per year")


table = top1000.pivot_table('prop', index = 'year', columns = 'sex', aggfunc = sum)
table.plot(title="Sum of table1000.prop by year and sex",
           yticks = np.linspace(0,1.2,13), xticks = range(1880, 2020, 10))
df = boys[boys.year== 2010]
df

prop_cumsum = df.sort_index(by='prop', ascending = False).prop.cumsum()
prop_cumsum[:10]
prop_cumsum.searchsorted(0.5)

df = boys[boys.year ==1900]

in1900 = df.sort_index(by='prop', ascending = False).prop.cumsum()

in1900.searchsorted(0.5)+1

def get_quantile_count(group, q = 0.5):
    group = group.sort_index(by='prop', ascending= False)
    return group.prop.cumsum().searchsorted(q)+1
diversity = top1000.groupby(['year','sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
diversity.head()
#Could not to plot???
#diversity.plot(title="Number of popular names in top 50%")
#Here the type of data should be numeric to plot
df = diversity.replace(',', '', regex=True)
df = df.replace('-', 'NaN', regex=True).astype('float')
df.plot()


get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'las_letter'

table = names.pivot_table('births', index = last_letters, columns = ['sex','year'], aggfunc= sum)

subtable = table.reindex(columns = [1910,1960,2010], level = 'year')

subtable.head()

subtable.sum()

letter_prop = subtable/subtable.sum().astype(float)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize = (10,8))
letter_prop['M'].plot(kind = 'bar', rot = 0, ax = axes[0], title = 'Male')
letter_prop['F'].plot(kind = 'bar', rot = 0, ax = axes[1], title = 'Famale', legend = False)


letter_prop = table/table.sum().astype(float)

dny_ts = letter_prop.ix[['d','n','y'], 'M'].T

dny_ts.head()
dny_ts.plot()


#iPython
import random
data = {i: random.random() for i in range(7)}
data

from numpy.random import randn
data = {i:randn() for i in range(7)}
data

an_apple = 27
an_example =42

b = [1,2,3]

import datetime
datetime.date
# %run PythonforDATest.py

def add_numbers(a,b):
    """
    Add two numbers together
    Returns
    :param a:
    :param b:
    :return:
    """
    return a+b
# ?add_numbers()
# ??add_numbers()

#Report the execution time of statement
# %time
# %timeit
# %who
# %who_ls
# %whos
#
# %logstart
# %logstate
# %logon
#
# %bookmark db C:/Users/mhan0
# %cd db

from numpy.random import randn
def add_and_sum(x,y):
    added =x +y
    summed = added.sum(axis=1)
    return summed
def call_function():
    x = randn(1000,1000)
    y = randn(1000,1000)
    return add_and_sum(x,y)

# %run prof_mod

# NumPy
import numpy as np
data = [[ 0.9526, -0.246 , -0.8856],[ 0.5639, 0.2379, 0.9104]]
data+data
data*10


data1 = [6,7.5,8,0,1]
arr1 = np.array(data1)

data2 = [[1,2,3,4],[5,6,7,8]]

arr2 = np.array(data2)
arr2

arr2.ndim
arr2.shape

arr1.dtype
arr2.dtype

np.zeros(10)

np.zeros((3,6))

np.empty((2,3,2))

np.arange(15)

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])

data = randn(7,4)

names
data

names == 'Bob'

data[names == 'Bob']

data[names == 'Bob',2:]

data[names == 'Bob',3]

names!='Bob'
data[-(names=='Bob')]
data

mask = (names == 'Bob')|(names =='Will')
mask

data[mask]

data[data<0] = 0
data

data[names != 'Joe'] = 7
data

arr = np.empty((8,4))
for i in range(8):
    arr[i] = i

arr
arr[[4,3,0,6]]
arr[[-3,-5,-7]]

points = np.arange(-5,5,0.01)
xs,ys = np.meshgrid(points,points)

import matplotlib as plt
# %pylab
z = np.sqrt(xs**2+ys**2)

z

plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title("Image plot of $\sqrt{x^2+y^2}$ for a grid of values")


import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

# plot(walk)
from pandas import Series, DataFrame
import pandas as pd

#Series
obj = Series([4,7,-5,3])

obj.values
obj.index

obj2 = Series([4,7,-5,3], index = ['d','b','a','c'])
obj2.index
obj2['a']
'b' in obj2

'e' in obj2

sdata = {'Ohio': 35000, 'Texas':71000, 'Oregon':16000,'Utah':5000}
obj3 = Series(sdata)
obj3

states = ['California','Ohio','Oregon','Texas']
obj4 = Series(sdata,index = states)

pd.isnull(obj4)
obj4.values

obj3+obj4


obj4.name = 'population'
obj4.index.name = 'state'
obj4

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)


df = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],index=['one', 'two', 'three', 'four', 'five'])
df['state']

df.ix['four']

df['debt'] = 16.5
df

df['xxx'] = df['debt']+df['pop']
df['xxx']

df['eastern'] = df['state']=='Ohio'
df.columns

pop = {'Nevada': {2001: 2.4, 2002: 2.9},'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)

frame3

frame3.T

frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3.values

x = pd.read_clipboard
x.__sizeof__()



from lxml.html import parse
from urllib import request
parsed = parse(request.urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
doc = parsed.getroot()
links = doc.findall('.//a')
