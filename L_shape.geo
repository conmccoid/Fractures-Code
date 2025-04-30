h=25;

//Points
Point(1) = {0,250,0,0.25};
Point(2) = {0,500,0,h};
Point(3) = {500,500,0,h};
Point(4) = {500,250,0,h};
Point(5) = {470,250,0,h};
Point(6) = {250,250,0,0.25};
Point(7) = {250,0,0,h};
Point(8) = {0,0,0,h};

//Lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,1};

Line Loop(1) = {1:8};

//Surface
Plane Surface(1)={1};
Physical Surface(1) = {1};
Physical Line(10) = {7};
Physical Point(100) = {5};