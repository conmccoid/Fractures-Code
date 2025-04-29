h=2.5;

//Points
Point(1) = {0,0,0,h};
Point(2) = {0,500,0,h};
Point(3) = {500,500,0,h};
Point(4) = {500,250,0,h};
Point(5) = {250,250,0,h};
Point(6) = {250,0,0,h};

//Lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,1};

Line Loop(1) = {1:6};

//Surface
Plane Surface(1)={1};
Physical Surface(1) = {1};
Physical Line(10) = {6};
Physical Line(20) = {1,2,3,6};