#include <cmath>
#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>

using std::cout;

const double sigma = 1;
double L, R;
const double eps = 0.1;
double a,b,c,d;         // interpolation constants for force
double a1,b1,c1,d1;     // interpolation constants for energy

// declaration of the atom data structure
struct Atom {
  double v[2];
  double x[2];
} ;

// initialisation of atom position and momentum
Atom InitialiseAtom(double x0, double x1, double v0, double v1) {
  
  Atom a;
  a.x[0] = x0; a.x[1] = x1;
  a.v[0] = v0; a.v[1] = v1;
  
  return a;
}

// function to compute x mod N
double modulo(double x,double N){
    return std::fmod( (std::fmod(x,N) + N) ,N);
}

// enforce the boundary conditions at the boundary of the domain
void enforce_boundary(std::vector<Atom>& atoms, double L, std::string boundary_type)
{
    if (boundary_type == "periodic") 
    {
        for (int i=0; i < atoms.size(); i++)
        {
            atoms[i].x[0] = modulo(atoms[i].x[0], L);
            atoms[i].x[1] = modulo(atoms[i].x[1], L);
        }
    }
    if (boundary_type == "reflective")
    {
        for (int i=0; i < atoms.size(); i++)
        {
            atoms[i].x[0] = modulo(L - std::fmod(atoms[i].x[0], L), L);
            atoms[i].x[1] = modulo(L - std::fmod(atoms[i].x[1], L), L);
        }
    }

}

// initialise atoms into a Cartesian grid
void initial_lattice(int size, double delta, std::vector<double> velocity, double L, std::vector<Atom>& atoms)
{
    double start_d=0;
    double x_top=0, x_bot=0, y_top=0, y_bot=0;

    if (size % 2 == 0)
    {
        start_d = 0.5 * delta + (size/2 - 1) * delta;
    }
    else
    {
        start_d = size * delta;
    }

    for (int i=0; i < size/2; i++)
    {
        x_top = L/2 - start_d;
        x_bot = L/2 + start_d;

        y_top = L/2 + start_d - i * delta;
        y_bot = L/2 - start_d + i * delta;

        for (int j=0; j < size; j++)
        {
            atoms.push_back(InitialiseAtom(x_top + j * delta, y_top, velocity[0], velocity[1]));
            atoms.push_back(InitialiseAtom(x_bot - j * delta, y_bot, velocity[0], velocity[1]));
        }
    }
    if (size % 2 != 0)
    {
        for (int j=0; j < size; j++)
        {
            atoms.push_back(InitialiseAtom(x_top + j * delta, L/2, velocity[0],velocity[1]));
        }
    }

}

// returns the Euclidean distance between atoms a,b in 2D
double sep(const Atom& a, const Atom& b)
{
  return pow(pow((a.x[0]-b.x[0]),2.0)+pow(a.x[1]-b.x[1],2.0),(0.5));
}

// return the cubic energy interpolation function
double energy_interpolation(double r){
  return a1*pow(r,3.0) + b1*pow(r,2.0) + c1*r + d1;
}

// return the cubic force interpolation function
double force_interpolation(double r){
  return a*pow(r,3.0) + b*pow(r,2.0) + c*r + d;
}

// returns LJ potential for interatomic distance r
double U_LJ(double r){
  
  if (r < R-eps) {
    return pow(sigma/r,12) - pow(sigma/r,6);
  }
  else {
    return energy_interpolation(r);
  }
  
}

// returns the derivative of LJ potential for interatomic distance r
double dUdr(double r){
  
  if (r < R-eps) {
    return -12*pow(sigma,12) / pow(r,13) + 6*pow(sigma,6) / pow(r,7);
  }
  else {
    return -1.0*force_interpolation(r);
  }
  
}

// returns total kinetic energy of all the atoms in vector
double kinetic(const std::vector<Atom> &atoms)
{ 
  double total_ke = 0.0;

  // sum kinetic energy over all atoms
  for (int i = 0; i < atoms.size(); i++)
  {
    total_ke += 0.5 * (pow(atoms[i].v[0], 2) + pow(atoms[i].v[1], 2));
  }

  return total_ke;
}

// returns PE between 2 atoms and forces on the 1st atom
std::vector<double> derivates_LJ(double r, const Atom& a, const Atom& b){
  
  std::vector<double> result(3);
  
  result[0] = U_LJ(r);   // potential energy
  
  double dU_dr = dUdr(r);
  
  double dnorm_dqx = (a.x[0]-b.x[0])/r;
  double dnorm_dqy = (a.x[1]-b.x[1])/r;
  
  result[1] = -1.0*dU_dr * dnorm_dqx;   //x force on a
  result[2] = -1.0*dU_dr * dnorm_dqy;   //y force on a
  
  //Note the forces on b is just the negative of forces on a
  
  return result;
  
}

// updates 2d forces_step array, where 0th row is x forces and 1st row is y forces
// also updates the potential energy on each atom at each time step
void forces(std::vector<Atom> &atoms, std::vector< std::vector<double> > &forces_step, double &energy) // sums all the forces acting on all atoms
{

  for (int i=0; i<atoms.size(); i++) {
    forces_step[0][i] = 0.0;
    forces_step[1][i] = 0.0;
  }
  
  bool hor, vert;

  //Atom temp is a temporary atom for horizontal, vertical and diagonal directions.
  std::vector<double> forces_temp(3);
  double r, dx, dy;
  
  for (int i = 0; i<atoms.size(); i++)
  {
    for (int j = i+1; j<atoms.size(); j++) {
      
        hor = 0; vert = 0;
      
        r = sep(atoms[i], atoms[j]);
        // if within range then add forces:
        if (r<R) {
          
          forces_temp = derivates_LJ(r,atoms[i],atoms[j]); // get PE between atoms i,j and x,y forces on atom a
          energy += forces_temp[0];
          forces_step[0][i] += forces_temp[1];
          forces_step[1][i] += forces_temp[2];
          forces_step[0][j] -= forces_temp[1];       //negative of x force on a
          forces_step[1][j] -= forces_temp[2];       //negative of y force on a
        }
        // if not within range, check border cells:
        else {

          // -------- HORIZONTAL border cells ----------
          // here find out the horizontal position of atom j wrt atom i, and if within cutoff distance, compute force

          // atom j is to the RIGHT of atom i
          if (atoms[j].x[0] > L-R && atoms[i].x[0] < R) {
            hor = 1;
            dx = (atoms[j].x[0]-L) - atoms[i].x[0];   //dist to copy of atom j in cell to left (negative)
            // if within range:
            r = pow(pow(dx,2.0)+pow(atoms[j].x[1]-atoms[i].x[1],2.0),0.5);
            if (r<R) {
              Atom temp = InitialiseAtom(atoms[j].x[0]-L, atoms[j].x[1], atoms[j].v[0], atoms[j].v[1]);
              forces_temp = derivates_LJ(r,atoms[i], temp);
              energy += forces_temp[0];
              forces_step[0][i] += forces_temp[1];
              forces_step[1][i] += forces_temp[2];
              forces_step[0][j] -= forces_temp[1];       //negative of x force on a
              forces_step[1][j] -= forces_temp[2];       //negative of y force on a

            }
            
          }
          // else atom j is to the LEFT of atom i
          if (atoms[i].x[0] > L-R && atoms[j].x[0] < R) {
            hor = 1;
            dx = (atoms[j].x[0]+L) - atoms[i].x[0];   //dist to copy of atom j in cell to right (positive)
            // if within range:
            r = pow(pow(dx,2.0)+pow(atoms[j].x[1]-atoms[i].x[1],2.0),0.5);
            if (r<R) {
              Atom temp = InitialiseAtom(atoms[j].x[0]+L, atoms[j].x[1], atoms[j].v[0], atoms[j].v[1]);
              forces_temp = derivates_LJ(r,atoms[i], temp);
              energy += forces_temp[0];
              forces_step[0][i] += forces_temp[1];
              forces_step[1][i] += forces_temp[2];
              forces_step[0][j] -= forces_temp[1];       //negative of x force on a
              forces_step[1][j] -= forces_temp[2];       //negative of y force on a

            }
            
          }

          // -------- VERTICAL border cells ----------

          // atom j is ABOVE atom i
          if (atoms[j].x[1] > L-R && atoms[i].x[1] < R) {
            vert = 1;
            dy = (atoms[j].x[1]-L) - atoms[i].x[1];   //dist to copy of atom j in cell below (negative)
            // if within range:
            r = pow(pow(dy,2.0)+pow(atoms[j].x[0]-atoms[i].x[0],2.0),0.5);
            if (r<R) {
              Atom temp = InitialiseAtom(atoms[j].x[0], atoms[j].x[1]-L, atoms[j].v[0], atoms[j].v[1]);
              forces_temp = derivates_LJ(r,atoms[i], temp);
              energy += forces_temp[0];
              forces_step[0][i] += forces_temp[1];
              forces_step[1][i] += forces_temp[2];
              forces_step[0][j] -= forces_temp[1];       //negative of x force on a
              forces_step[1][j] -= forces_temp[2];       //negative of y force on a

            }
            
          }
          // else atom j is BELOW atom i
          if (atoms[i].x[1] > L-R && atoms[j].x[1] < R) {
            vert = 1;
            dy = (atoms[j].x[1]+L) - atoms[i].x[1];   //dist to copy of atom j in cell above (positive)
            // if within range:
            r = pow(pow(dy,2.0)+pow(atoms[j].x[0]-atoms[i].x[0],2.0),0.5);
            if (r<R) {
              Atom temp = InitialiseAtom(atoms[j].x[0], atoms[j].x[1]+L, atoms[j].v[0], atoms[j].v[1]);
              forces_temp = derivates_LJ(r,atoms[i], temp);
              energy += forces_temp[0];
              forces_step[0][i] += forces_temp[1];
              forces_step[1][i] += forces_temp[2];
              forces_step[0][j] -= forces_temp[1];       //negative of x force on a
              forces_step[1][j] -= forces_temp[2];       //negative of y force on a
 
            }
            
          }
          // -------- DIAGONAL border cells ----------
          // uses values of dx and dy computed earlier (so we know the location of atom j wrt atom i)
          // if within range:
          if (hor && vert) {
            r = pow(pow(dx,2.0)+pow(dy,2.0),0.5);
            if (r<R) {
              Atom temp = InitialiseAtom(atoms[i].x[0]+dx, atoms[i].x[1]+dy, atoms[j].v[0], atoms[j].v[1]);
              forces_temp = derivates_LJ(r,atoms[i], temp);
              energy += forces_temp[0];
              forces_step[0][i] += forces_temp[1];
              forces_step[1][i] += forces_temp[2];
              forces_step[0][j] -= forces_temp[1];       //negative of x force on a
              forces_step[1][j] -= forces_temp[2];       //negative of y force on a

            }
          }
        }
    }
  }

}

// Implements velocity Verlet as an integrator
void verlet(std::vector<Atom> &atoms, std::vector< std::vector<double> > &forces_step, double &energy, double h, double L, std::string boundary_type = "periodic")
{
    double kin_en = 0.0;
    
    // half step velocity with exisiting forces
    for (int i=0; i < atoms.size(); i++)
    {
        atoms[i].v[0] += forces_step[0][i] * h/2.0;
        atoms[i].v[1] += forces_step[1][i] * h/2.0;
    }

    // full position step
    for (int i=0; i < atoms.size(); i++)
    {
        atoms[i].x[0] += atoms[i].v[0] * h;
        atoms[i].x[1] += atoms[i].v[1] * h;
    }        

    enforce_boundary(atoms, L, boundary_type);        // Boundary Conditions
    forces(atoms, forces_step, energy);               // calculate forces

    // second half velocity step with new forces
    for (int i=0; i < atoms.size(); i++)
    {
        atoms[i].v[0] += forces_step[0][i] * h/2.0;
        atoms[i].v[1] += forces_step[1][i] * h/2.0;
    }
  
    kin_en = kinetic(atoms);
    energy += kin_en;
}


int main()
{
  // --------------- Start Timing and Timing Variables ------------------
  
  double writingT = 0;
  
  auto start = std::chrono::high_resolution_clock::now();
  
  // -------------- Initialising Parameters----------------
  
  // physical parameters
  
  int lat_size = 16;                 // Number of atoms per side of domain
  int Ntot = pow(lat_size,2);       // Total number of atoms in the lattice
  double delta_init = pow(2,1.0/6.0)*sigma;    // initial atom separation
  L = delta_init * lat_size;        // Length of the entire domain
  R = 2.5 * sigma;                  // Cut off radius for potential
  std::vector<double> init_velocity(2, 0);     // initial velocity to use for the atoms (set to 0)
  
  // numerical parameters
  
  double t = 0.0; double tf = 50;       // start and finish time
  int n = 500000;                       // number of verlet steps
  double step = (tf-t)/double(n);       // time step size

  std::string boundary_type="periodic"; // Type of boundary used (periodic or reflective)
  std::string write_pos="True";         // option to turn off writing positions at each step (True or False)
 
  cout << "L = " << L << std::endl;
  cout << "delta_init = " << delta_init << std::endl;
  cout << "R = " << R << std::endl;
  cout << "Number of atoms = " << Ntot << std::endl;
  
  //------------ Interpolation Coefficients --------------
  
  a = (6.0 * pow(sigma,6.0) * ((9.0*eps - 2.0*R) * pow(eps - R,6.0) + 2.0 * (-15.0*eps + 2.0*R)* pow(sigma,6.0)))/(pow(eps,3.0)*pow(eps - R,14.0));
  
  b = (12.0*pow(sigma,6.0) * (pow(eps - R,6.0) * (5.0 * pow(eps,2.0) - 15.0*eps*R + 3.0*pow(R,2.0)) - 2.0 * (8*pow(eps,2.0) - 24*eps*R + 3.0*pow(R,2.0)) * pow(sigma,6.0)))/(pow(eps,3.0)*pow(eps - R,14.0));
  
  c = 6.0*R*pow(sigma,6.0)*(-pow(eps - R,6.0) * (20.0 * pow(eps,2.0) - 33.0*eps*R + 6.0*pow(R,2.0)) + 2.0 * (32.0 * pow(eps,2.0) - 51.0*eps*R + 6.0*pow(R,2.0))* pow(sigma,6.0))/(pow(eps,3.0)*pow(eps - R,14.0));
  
  d = (12.0*pow(R,2.0)*pow(sigma,6.0) * (pow(eps - R,6.0)*(5.0*eps - R) + 2.0 * (-8.0*eps + R) * pow(sigma,6.0)))/(pow(eps,3.0)*pow(eps - R,13.0));
  
  a1 = (-2.0*pow(eps-R,6.0)*(4.0*eps-R)*pow(sigma,6.0) + 2.0*(7.0*eps-R)*pow(sigma,12.0))/(pow(eps,3.0)*pow(eps - R,13.0));
  
  b1 = (-3.0*pow(eps-R,6.0)*(3.0 * pow(eps,2.0) - 9.0*eps*R + 2.0*pow(R,2.0))*pow(sigma,6.0) + 3.0*(5.0 * pow(eps,2.0) - 15.0*eps*R + 2.0*pow(R,2.0))*pow(sigma,12.0))/(pow(eps,3.0)*pow(eps - R,13.0));
  
  c1 = 6.0*R*pow(sigma,6.0)*(pow(eps-R,6.0)*(3.0 * pow(eps,2.0) - 5.0*eps*R + pow(R,2.0)) - (5.0 * pow(eps,2.0) - 8.0*eps*R + pow(R,2.0))*pow(sigma,6.0))/(pow(eps,3.0)*pow(eps - R,13.0));
  
  d1 = pow(R,2.0)*pow(sigma,6.0)*(pow(eps-R,6.0)*(-9.0*eps+2.0*R) + (15.0*eps-2.0*R)*pow(sigma,6.0))/(pow(eps,3.0)*pow(eps - R,12.0));
  
  // ----------- Initialise the lattice and forces vector ------------
  
  std::vector<Atom> atoms;              // vector of atoms
  
  initial_lattice(lat_size, delta_init, init_velocity, L, atoms);
  
  for (int i=0; i<Ntot; i++) {
    if (atoms[i].x[0] < L/2) {
      atoms[i].v[0] += 0.001;
    }
    else {
      atoms[i].v[0] -= 0.001;
    }
    if (atoms[i].x[1] < L/2) {
      atoms[i].v[1] += 0.001;
    }
    else {
      atoms[i].v[1] -= 0.001;
    }
  }

  std::ofstream myfile_energy;
  std::ofstream myfile_pos;
  std::ofstream myfile_vel;
  
  myfile_energy.open ("energy_ser.dat");
  if (write_pos=="True") {
    myfile_pos.open ("pos_ser.dat");
    myfile_vel.open ("vel_ser.dat");
  }
  
  double E = 0.0;             // initialise energy variable here

  cout << "----------" << std::endl;
  cout << "Number of Steps: " << n << std::endl;
  cout << "----------" << std::endl;
  
  std::vector< std::vector <double> > forces_step;
  std::vector<double> forces_x(atoms.size());
  forces_step.push_back(forces_x);
  forces_step.push_back(forces_x);
  
  for (int i=0; i<atoms.size(); i++) {
    forces_step[0][i] = 0.0;
    forces_step[1][i] = 0.0;
  }
  
  forces(atoms,forces_step, E);  // inital force caluclation
  
  // ---------------------------- MAIN LOOP ---------------------------------------------

  for (int i = 0; i<n; i++)
  {
    if (i%10000==0) {
      cout << "Iteration " << i << std::endl;
    }
    
    verlet(atoms, forces_step, E, step, L, boundary_type);  // Take a verlet step
    
    //  Print positions at each step to file
    // time printing separately
    
    auto Writingstart = std::chrono::high_resolution_clock::now();
    
    if (write_pos == "True")
    {
      for (int j = 0; j < Ntot; j++) {
        myfile_pos << atoms[j].x[0] << " " << atoms[j].x[1] << " ";
        myfile_vel << atoms[j].v[0] << " " << atoms[j].v[1] << " ";
      }
      myfile_pos << std::endl;
      myfile_vel << std::endl;
    }
    
    myfile_energy << E << std::endl;
    auto Writingstop = std::chrono::high_resolution_clock::now();
    auto Writingduration = std::chrono::duration_cast<std::chrono::microseconds>(Writingstop - Writingstart);
    writingT += Writingduration.count();
    
    E = 0.0; // set energy to zero for next timestep
  
  }
  // --------------------------MAIN LOOP ENDS -------------------------------------------
  
  
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  cout << "Time taken for " << Ntot << " atoms: "  << duration.count()*0.000001 << " seconds" << std::endl;
  cout << "Time taken for writing the data: " << double(writingT)*0.000001 << " seconds" << std::endl;
  
  // Close the output files
  myfile_energy.close();
  if (write_pos=="True") {
    myfile_pos.close();
    myfile_vel.close();
  }
        
  return 0;
}
