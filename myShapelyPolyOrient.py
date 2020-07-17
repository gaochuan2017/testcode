import shapely.geometry as shg
import numpy as np

if __name__ == "__main__":
    l=[(0,0),(100,0),(100,100),(0,100)]
#    l=[(0,0),(100,100),(100,0),(0,100)]
    p1=shg.Polygon(l) 
    print(p1)
    p2=shg.polygon.orient(p1,sign=-1)#返回一个给定多边形正确方向多边形，1顺时针，-1逆时针
    print(p2)
#Polygon.exterior.coords: 外部轮廓的点
    print(list(p2.exterior.coords))
