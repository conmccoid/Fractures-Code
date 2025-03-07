//Définition des valeurs physiques
L = 65; //Longeur selon x
H = 120; //Longuer selon y
y0 = 0;  // y-coord of teh initial crack
l0 = 10; // Length initial crack
eps = 0.01; // Width of initial crack
h = 1; //0.25; //0.1; //0.3; //1.5; //2 //Taille du maillage
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

//Final Surface
Plane Surface(1) = {1};


//Outputs
Physical Surface(1) = {1};
Physical Line(4000) = {6,7};
Physical Line(2000) = {1:5};