import os
import codecs
'''
with codecs.open("naxiekeng.txt",'r+',encoding='gb2312') as f:
    lines=f.readlines()
    for line in lines:
        print(line)
'''
with open("aaa.txt",'r') as f:
    lines = f.readlines()
    if(not len(lines)):
        print("empty file!")
    else:
        print(lines)
    #line=lines[0]
    #print(line)