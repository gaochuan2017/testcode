import os
cur_dir=os.getcwd()
#os.removedirs("a")
if not(os.path.exists("a")):
    os.makedirs("a")
print(os.listdir())
for (d1,d2,d3) in os.walk(cur_dir+os.sep+'a'):
    print(d1)
    print(d2)
    print(d3)
