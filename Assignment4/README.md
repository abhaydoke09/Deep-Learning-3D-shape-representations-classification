# 3D Surface reconstruction from point clouds

#### In this assignment I implemented three implicit surface reconstruction algorithms to approximate a surface represented by scattered point data. 

##### The problem can be stated as follows:
###### Given a set of points **P = {p<sub>1</sub>,p<sub>2</sub>,...,p<sub>n</sub>}** in a point cloud, we will define an implicit function f(x,y,z) that measures the signed distance to the surface approximated by these points. The surface is extracted at f(x,y,z) = 0 using the marching cubes algorithm.

#### Reconstruction Results

**1. Signed distance to tangent plane**
<img src="./imgs/1.png" width="200" height="200" />

**2. Moving least squares distance to tangent** 
<img src="./imgs/2.png" width="200" height="200" />

**3. Radial Basis Function (RBF) interpolation for approximating the signed distance**
<img src="./imgs/3.png" width="200" height="200" />
