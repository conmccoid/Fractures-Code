//Définition des valeurs physiques
L = 65; //Longeur selon x
H = 120; //Longuer selon y
y0 = 5;  // y-coord of teh initial crack
l0 = 10; // Length initial crack
eps = 0.01; // Width of initial crack
h = 0.05; //0.1; //0.3; //1.5; //2 //Taille du maillage
hc = 1.;

//Plate
Point(1) = {0,y0-eps,0,h};
Point(2) = {0,-H/2,0,hc};
Point(3) = {L,-H/2,0,hc};
Point(4) = {L,H/2,0,hc};
Point(5) = {0,H/2,0,hc};
Point(6) = {0,y0+eps,0,h};
Point(100) = {l0,y0,0,h};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,100};
Line(7) = {100,1};

Line Loop(1) = {1:7};

//Hole 1
radius = 5;
Point(7) = {20, H/2-20, 0, h}; //Centre of the hole
Point(8) = {20+radius, H/2-20, 0, hc};
Point(9) = {20-radius, H/2-20, 0, hc};
Point(10) = {20, H/2-20+radius, 0, hc};
Point(11) = {20, H/2-20-radius, 0, hc};

Circle(17) = {10, 7, 9};
Circle(18) = {9, 7, 11};
Circle(19) = {11, 7, 8};
Circle(20) = {8, 7, 10};
Line Loop(2) = {17:20};

//Hole 2
Point(12) = {20, -H/2+20, 0, h}; //Centre of the hole
Point(13) = {20+radius, -H/2+20, 0, hc};
Point(14) = {20-radius, -H/2+20, 0, hc};
Point(15) = {20, -H/2+20+radius, 0, hc};
Point(16) = {20, -H/2+20-radius, 0, hc};

Circle(21) = {15, 12, 14};
Circle(22) = {14, 12, 16};
Circle(23) = {16, 12, 13};
Circle(24) = {13, 12, 15};
Line Loop(3) = {21:24};

//Hole 3
radius = 10;
Point(37) = {L-28.5, -H/2+51, 0, h}; //Centre of the hole
Point(38) = {L-28.5+radius, -H/2+51, 0, h};
Point(39) = {L-28.5-radius, -H/2+51, 0, h};
Point(40) = {L-28.5, -H/2+51+radius, 0, h};
Point(41) = {L-28.5, -H/2+51-radius, 0, h};

Circle(55) = {40, 37, 39};
Circle(56) = {39, 37, 41};
Circle(57) = {41, 37, 38};
Circle(58) = {38, 37, 40};
Line Loop(4) = {55:58};

//Final Surface
Plane Surface(1) = {1, 2, 3, 4};


//Outputs
Physical Surface(1) = {1};
Physical Line(20) = {17:20};
Physical Line(30) = {21:24};
Physical Line(40) = {6,7};
