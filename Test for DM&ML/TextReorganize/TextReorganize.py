__author__ = 'mhan7'

Fulllist = []
AttendList = []

FulllistFile = open("TextReorganize\TPCFullList.txt", "r",encoding = 'ISO-8859-1')
for line in FulllistFile:
    line = line.encode("utf-8",'replace')
    print(line)
    Fulllist.append(line)
FulllistFile.close()
len(Fulllist)

AttendListFile = open("TextReorganize\TPCList.txt", "r",encoding = 'ISO-8859-1')
for line in AttendListFile:
    line = line.encode("utf-8",'replace')
    print(line)
    AttendList.append(line)
AttendListFile.close()
len(AttendList)

ResultList = []
for AttendName in AttendList:
    for Name in Fulllist:
        if(AttendName[:-2] in Name):

            ResultList.append(Name.decode("gbk")[:-2].replace('(','\t'))
            #print("TRUE"+Name)
        #else:
           # print(AttendName.decode("utf-8",'replace'))
len(ResultList)
for r in ResultList:
    print(r)
RFile = open("Result.txt", 'w')
for r in ResultList:
    RFile.write(r)
    RFile.write('\n')

RFile.close()

str = "Yong Cui (Tsinghua University, P.R. China)"

import string

print(str[:-1].replace('(','\t'))
open()

AttendName[:-2]




for AttendName in AttendList:
    for Name in Fulllist:
        if(AttendName[:-2] in Name):
            ResultList.append(Name)
            print("TRUE"+ Name)
        else:
            print(AttendName)

len(ResultList)

"abc" in "abcdef"
a = AttendList[0]
b = Fulllist[0]
a[:-2] in b
b
a[:-2] in b

"abc\n" in "abc"