import sys
import codecs
print(sys.maxunicode)
'''
for more about unicode,utf-8,please click:
    https://blog.csdn.net/wsl_cnxw/article/details/82083743
'''

a="风卷残云"

print(type(a))
'''
print(a)
b=a.encode("utf-8")

print(type(b))
print(b)
'''
'''
with codecs.open("afile.txt",'w',encoding='utf-8') as f:
    f.write(a)

#f1=open("~/testcode/aflie.txt",'r')
with codecs.open("afile.txt",'r',encoding='utf-8') as f:
    print(f.readline())
'''
with open("bfile.txt",'w+') as f:
    f.write('32')

with open("bfile.txt",'r') as f:
    print(f.readline())
with codecs.open("afile.txt",mode='w',encoding='utf-8') as f:
    f.write("袁崇焕")
print("write chinse refers to utf-8,let's try to read it from afile.txt")
with open("afile.txt",'r') as f:
    print(f.readline())

print(type(a)==type(u"风卷残云"))

a1=u"风卷残云".encode('utf-8')
print("风卷残云 stored in utf-8 forms, ",a1)
print("风卷残云 stored in unicode forms, ",u"风卷残云")

