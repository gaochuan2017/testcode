import os
import codecs
with codecs.open("naxiekeng.txt",'r+',encoding='gb2312') as f:
    lines=f.readlines()
    for line in lines:
        print(line)
    
    #line=lines[0]
    #print(line)