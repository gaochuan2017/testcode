import os
if __name__ == "__main__":
    fulname='/home/gaochuan/testcode/basename.py'
    [print(a1) for a1 in os.path.split(fulname)]
    print("basename is ",os.path.basename(fulname))
    pass
