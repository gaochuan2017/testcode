import os
if __name__ == "__main__":
    fulname='/home/gaochuan/testcode/basename.py'
    [print(a1) for a1 in os.path.split(fulname)]
    a,b=os.path.split(fulname)
    print(a,b)
    print("#####",os.path.splitext(fulname))
    print("basename is ",os.path.basename(fulname))
# " ".join(str): add " " into str.
    a2=' '.join(fulname.split(sep=os.sep))
    print(a2)

    
    pass
