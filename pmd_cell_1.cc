#include <cmath>
#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <mpi.h>
#include <algorithm>
using std::cout;

const double sigma = 1;
double L, R;
int row, col;
const double eps = 0.1;
int rank, nproc;
double a,b,c,d;         // interpolation constants for force
double a1,b1,c1,d1;     // interpolation constants for energy

MPI_Comm comm;
MPI_Datatype MPI_atom;

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
double modulo(double x,double N) {
    return std::fmod( (std::fmod(x,N) + N) ,N);
}

// gathers all the atoms to root and then prints to two files
void print(std::vector<Atom>& atoms, std::ofstream& myfile_pos, std::ofstream& myfile_vel, int nproc, int Ntot) {
  
  std::vector<int> natoms_vec;
  std::vector<Atom> atoms_root;
  std::vector<int> displace;
  int n_atoms = atoms.size();
  
  if (rank==0) {
    natoms_vec.resize(nproc);
    atoms_root.resize(Ntot);
    displace.resize(nproc);
  }
  // get the number of atoms in each process to rank 0
  MPI_Gather(&(n_atoms), 1, MPI_INT, &(natoms_vec[0]), 1, MPI_INT, 0, comm);
  
  // create displacement vector for gatherv (each process sends a different number of atoms), all atoms collected to atoms_root vector
  if (rank==0) {
    displace[0] = 0;
    for (int i=1; i<nproc; i++) {
      displace[i] = displace[i-1] + natoms_vec[i-1];
    }
  }
  
  MPI_Gatherv(&(atoms[0]), atoms.size(), MPI_atom, &(atoms_root[0]), natoms_vec.data(), displace.data(), MPI_atom, 0, comm);

  if (rank == 0) {
    for (int j = 0; j < Ntot; j++) {
      myfile_pos << atoms_root[j].x[0] << " " << atoms_root[j].x[1] << " ";
      myfile_vel << atoms_root[j].v[0] << " " << atoms_root[j].v[1] << " ";
    }
    myfile_pos << std::endl;
    myfile_vel << std::endl;
  }
  
}

// enforce both processor boundaries and cell boundaries in this function
// send in x direction (across vertical process boundaries) first, and then in y direction to ensure diagonal are taken care of too.
// reflective boundary conditions need to be applied before sending, periodic boundary conditions need to be applied after sending
// first count how many atoms need to be sent in horizontal direction
// send and receive these ints in each direction
// send actual atoms in x direction
//     broke it down into even and odd (row/col) ranked processes so there's less deadlock waiting for synchronous sending
// then count how many atoms need to be sent in vertical direction
// then send actual atoms in y direction
void parallel_enforce_boundary(std::vector<Atom>& atoms, double L, double R, int sqnproc, std::string boundary_type) {
  // row is the process's row index
  // col is the process's column index
 
  // should do reflective boundary conditions first due to if statements
  if (boundary_type == "reflective") {
    for (int i=0; i < atoms.size(); i++) {
      atoms[i].x[0] = modulo(L - std::fmod(atoms[i].x[0], L), L);
      atoms[i].x[1] = modulo(L - std::fmod(atoms[i].x[1], L), L);
    }
  }
  
  // ----------------- Send x direction - same row, col+-1 -------------------------------------------------
  // first communicate the number of atoms that will be sent across vertical borders
  
  // count how atoms have to be sent. Naming convention : n_send_left = number to be sent to the left.
  //                                                      n_recv_left = number to be received from the left.
  int n_send_left = 0; int n_send_right = 0;
  int n_recv_left = 0; int n_recv_right = 0;
  std::vector<int> ind_send_left, ind_send_right;
  
  // checking how many and which atoms crossed horizontally
  for (int i=0; i<atoms.size(); i++) {
    if (atoms[i].x[0] > (col+1)*R) {
      n_send_right += 1;
      ind_send_right.push_back(i);
    }
    else if (atoms[i].x[0] < (col)*R) {
      n_send_left += 1;
      ind_send_left.push_back(i);
    }
  }

  // first send how many atoms will be sent later
  if (col%2==0) {   // send from even columns 
    // send&recv right
    MPI_Ssend(&n_send_right, 1, MPI_INT, row*sqnproc+modulo(col+1,sqnproc), row, comm);
    MPI_Recv(&n_recv_left, 1, MPI_INT, row*sqnproc+modulo(col-1,sqnproc), row, comm, MPI_STATUS_IGNORE);
    // send&recv left
    MPI_Ssend(&n_send_left, 1, MPI_INT, row*sqnproc+modulo(col-1,sqnproc), row, comm);
    MPI_Recv(&n_recv_right, 1, MPI_INT, row*sqnproc+modulo(col+1,sqnproc), row, comm, MPI_STATUS_IGNORE);
    

  } else {  // send from odd columns
    // recv&send right
    MPI_Recv(&n_recv_left, 1, MPI_INT, row*sqnproc+modulo(col-1,sqnproc), row, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&n_send_right, 1, MPI_INT, row*sqnproc+modulo(col+1,sqnproc), row, comm);
    // recv&send left
    MPI_Recv(&n_recv_right, 1, MPI_INT, row*sqnproc+modulo(col+1,sqnproc), row, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&n_send_left, 1, MPI_INT, row*sqnproc+modulo(col-1,sqnproc), row, comm);
  }

  // now send the atoms that crossed process boundaries - do this first due to following if statements
  if (col%2==0) {   // send right from even columns first then from odd columns. Then send left
    
    for (int i=0; i < n_send_right; i++) {
      MPI_Ssend(&(atoms[ind_send_right[i]]), 1, MPI_atom, row*sqnproc+modulo(col+1,sqnproc), i, comm);
    }
    for (int i=0; i < n_recv_left; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, row*sqnproc+modulo(col-1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      atoms.push_back(temp);
    }
    
    for (int i=0; i < n_send_left; i++) {
      MPI_Ssend(&(atoms[ind_send_left[i]]), 1, MPI_atom, row*sqnproc+modulo(col-1,sqnproc), i, comm);
    }
    for (int i=0; i < n_recv_right; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, row*sqnproc+modulo(col+1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      atoms.push_back(temp);
    }
    
    // now drop all the atoms sent all at the same time => concatenate and sort the ind_send vectors
    ind_send_right.insert( ind_send_right.end(), ind_send_left.begin(), ind_send_left.end() );
    std::sort(ind_send_right.begin(), ind_send_right.end() );
    for (int i=ind_send_right.size()-1; i >= 0 ; i--) {
      atoms.erase(atoms.begin()+ind_send_right[i]);
    }
    

  } else { // repeat for odd columns
    
    for (int i=0; i < n_recv_left; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, row*sqnproc+modulo(col-1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      atoms.push_back(temp);
    }
    for (int i=0; i < n_send_right; i++) {
      MPI_Ssend(&(atoms[ind_send_right[i]]), 1, MPI_atom, row*sqnproc+modulo(col+1,sqnproc), i, comm);
    }
    
    for (int i=0; i < n_recv_right; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, row*sqnproc+modulo(col+1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      atoms.push_back(temp);
    }
    for (int i=0; i < n_send_left; i++) {
      MPI_Ssend(&(atoms[ind_send_left[i]]), 1, MPI_atom, row*sqnproc+modulo(col-1,sqnproc), i, comm);
    }
    
    // now drop all the atoms sent all at the same time => concatenate the ind_send vectors
    ind_send_right.insert( ind_send_right.end(), ind_send_left.begin(), ind_send_left.end() );
    std::sort(ind_send_right.begin(), ind_send_right.end() );
    for (int i=ind_send_right.size()-1; i >= 0 ; i--) {
      atoms.erase(atoms.begin()+ind_send_right[i]);
    }

    
  }

  MPI_Barrier(comm);
  
  //---------------------- Send y direction - same col, row+-1 ---------------------------------------------
  
  // now communicate the number of atoms that will be sent across horizontal borders
  
  // count how atoms have to be sent
  int n_send_up = 0; int n_send_down = 0;
  int n_recv_up = 0; int n_recv_down = 0;
  std::vector<int> ind_send_up, ind_send_down;
  
  for (int i=0; i<atoms.size(); i++) {
    if (atoms[i].x[1] > (row+1)*R) {
      n_send_up += 1;
      ind_send_up.push_back(i);
    }
    else if (atoms[i].x[1] < (row)*R) {
      n_send_down += 1;
      ind_send_down.push_back(i);
    }
  }

  if (row%2==0) {   // send up from even row first then from odd rows. Then send down
    
    MPI_Ssend(&n_send_up, 1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+col, col, comm);
    MPI_Recv(&n_recv_down, 1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+col, col, comm, MPI_STATUS_IGNORE);
    
    MPI_Ssend(&n_send_down, 1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+col, col, comm);
    MPI_Recv(&n_recv_up, 1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+col, col, comm, MPI_STATUS_IGNORE);
    

  } else {
    
    MPI_Recv(&n_recv_down, 1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+col, col, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&n_send_up, 1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+col, col, comm);
    
    MPI_Recv(&n_recv_up, 1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+col, col, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&n_send_down, 1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+col, col, comm);
  }

  // now actually sending the atoms that crossed process boundaries - do this first due to following if statements
  
  if (row%2==0) {   // send up from even rows first then from odd rows. Then send down
    
    for (int i=0; i < n_send_up; i++) {
      MPI_Ssend(&(atoms[ind_send_up[i]]), 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+col, i, comm);
    }
    for (int i=0; i < n_recv_down; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+col, i, comm, MPI_STATUS_IGNORE);
      atoms.push_back(temp);
    }
    
    for (int i=0; i < n_send_down; i++) {
      MPI_Ssend(&(atoms[ind_send_down[i]]), 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+col, i, comm);
    }
    for (int i=0; i < n_recv_up; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+col, i, comm, MPI_STATUS_IGNORE);
      atoms.push_back(temp);
    }
    
    // now drop all the atoms sent all at the same time => concatenate and sort the ind_send vectors
    ind_send_up.insert( ind_send_up.end(), ind_send_down.begin(), ind_send_down.end() );
    std::sort(ind_send_up.begin(), ind_send_up.end() );
    for (int i=ind_send_up.size()-1; i >= 0 ; i--) {
      atoms.erase(atoms.begin()+ind_send_up[i]);
    }
    

  } else {
    
    for (int i=0; i < n_recv_down; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+col, i, comm, MPI_STATUS_IGNORE);
      atoms.push_back(temp);
    }
    for (int i=0; i < n_send_up; i++) {
      MPI_Ssend(&(atoms[ind_send_up[i]]), 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+col, i, comm);
    }
    
    for (int i=0; i < n_recv_up; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+col, i, comm, MPI_STATUS_IGNORE);
      atoms.push_back(temp);
    }
    for (int i=0; i < n_send_down; i++) {
      MPI_Ssend(&(atoms[ind_send_down[i]]), 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+col, i, comm);
    }
    
    // now drop all the atoms sent all at the same time => concatenate the ind_send vectors
    ind_send_up.insert( ind_send_up.end(), ind_send_down.begin(), ind_send_down.end() );
    std::sort(ind_send_up.begin(), ind_send_up.end() );
    for (int i=ind_send_up.size()-1; i >= 0 ; i--) {
      atoms.erase(atoms.begin()+ind_send_up[i]);
    }
    
  }

  // now impose periodic boundary conditions
  if (boundary_type == "periodic") {
    for (int i=0; i < atoms.size(); i++) {
      atoms[i].x[0] = modulo(atoms[i].x[0], L);
      atoms[i].x[1] = modulo(atoms[i].x[1], L);
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
double kinetic(const std::vector<Atom>& atoms){ 

  double total_ke = 0.0;  
  // sum kinetic energy over all atoms
  for (int i = 0; i < atoms.size(); i++)
  {
    total_ke += 0.5 * (pow(atoms[i].v[0], 2) + pow(atoms[i].v[1], 2));
  }

  return total_ke; //kinetic_vec;
}

// returns PE between 2 atoms and forces on the 1st atom
std::vector<double> derivatives_LJ(double r, const Atom& a, const Atom& b){
  
  std::vector<double> result(3);    // vector of potential energy and forces to return
  
  result[0] = U_LJ(r);   // potential energy
  
  // derivative of the potential = negative force (du/dx = du/dr * dr/dx) - same for du/dy
  double dU_dr = dUdr(r);                   // du/dr from above
  
  double dnorm_dqx = (a.x[0]-b.x[0])/r;     // dr/dx from above 
  double dnorm_dqy = (a.x[1]-b.x[1])/r;     // dr/dy
  
  result[1] = -1.0*dU_dr * dnorm_dqx;   // x force on a
  result[2] = -1.0*dU_dr * dnorm_dqy;   // y force on a
  
  //Note the forces on b is just the negative of forces on a
  
  return result;
  
}

// give it a list of indices of atoms and the full list of atoms in cell, list of atoms from neighbouring cell, update the cell_forces vector
void skin_forces_indices_atoms(const std::vector<int>& cell_skin_ind, const std::vector<Atom>& cell_atoms, const std::vector<Atom>& neighbour_atoms, std::vector< std::vector<double> >& cell_forces, double& temp_potential){
  
  std::vector<double> forces_temp(3);
  double r;
  
  for (int i=0; i<cell_skin_ind.size(); i++)     // loop over atoms within the cell skin
  {
    for (int j=0; j<neighbour_atoms.size(); j++) { // loop over atoms in neighbour skin
      r = sep(cell_atoms[cell_skin_ind[i]], neighbour_atoms[j]);
      // if within range then add forces:
      if (r < R) {
        
        forces_temp = derivatives_LJ(r,cell_atoms[cell_skin_ind[i]],neighbour_atoms[j]); // get PE between atoms i,j and x,y forces on atom a
        
        temp_potential += 0.5*forces_temp[0];
        
        cell_forces[0][cell_skin_ind[i]] += forces_temp[1];
        cell_forces[1][cell_skin_ind[i]] += forces_temp[2];
        
      }
    }
    
  }
  
}

// return the 2d forces vector acting on the atoms within the processor's cell
std::vector< std::vector<double> > update_force(const std::vector<Atom> &atoms, std::vector<double> &cell_energy, double R, double M, double sqnproc)
{

  int num_atoms = (int)atoms.size();         // number of atoms in the cell
  
  std::vector< std::vector<double> > cell_forces(2);
  cell_forces[0].resize(num_atoms);
  cell_forces[1].resize(num_atoms);
  
  for (int i=0; i<num_atoms; i++) {
    cell_forces[0][i] = 0.0;
    cell_forces[1][i] = 0.0;
  }
  
  std::vector<double> forces_temp(3);
  double r;
  double temp_potential = 0;                // initialise the potential at this step to 0
  
  // first figure out the 8 skin vectors - need all 8 including corners for diagonal sending
  
  std::vector<Atom> recv_left;     // atoms received in all directions
  std::vector<Atom> recv_right;
  std::vector<Atom> recv_up;
  std::vector<Atom> recv_down;
  std::vector<Atom> recv_upright;
  std::vector<Atom> recv_upleft;
  std::vector<Atom> recv_downright;
  std::vector<Atom> recv_downleft;

  std::vector<int> ind_left;      // indices of atoms sent
  std::vector<int> ind_right;
  std::vector<int> ind_up;
  std::vector<int> ind_down;
  std::vector<int> ind_upright; 
  std::vector<int> ind_upleft;
  std::vector<int> ind_downright;
  std::vector<int> ind_downleft;

  int n_recv_right, n_recv_left, n_recv_up, n_recv_down;
  int n_send_right, n_send_left, n_send_up, n_send_down;
  int n_recv_upright, n_recv_upleft, n_recv_downright, n_recv_downleft;
  int n_send_upright, n_send_upleft, n_send_downright, n_send_downleft;
  
  // assign the indices of the atoms which lie in the skins of the cell

  for (int i=0; i<num_atoms; i++) {
    if (atoms[i].x[0] < col*M + R) {
      ind_left.push_back(i);

      if (atoms[i].x[1] < row*M + R) 
      {
          ind_downleft.push_back(i);
      }
      else if (atoms[i].x[1] > (row+1)*M - R) 
      {
          ind_upleft.push_back(i);
      }
    }
    
    if (atoms[i].x[0] > (col+1)*M - R) {
      ind_right.push_back(i);

      if (atoms[i].x[1] < row*M + R) 
      {
          ind_downright.push_back(i);
      }
      else if (atoms[i].x[1] > (row+1)*M - R) 
      {
          ind_upright.push_back(i);
      }

    }
    
    if (atoms[i].x[1] < row*M + R) {
      ind_down.push_back(i);
    }
    
    if (atoms[i].x[1] > (row+1)*M - R) {
      ind_up.push_back(i);
    }

  }
  
  n_send_up         = ind_up.size();
  n_send_down       = ind_down.size();  // so close to perfection
  n_send_left       = ind_left.size();
  n_send_right      = ind_right.size();
  n_send_upleft     = ind_upleft.size();
  n_send_upright    = ind_upright.size();
  n_send_downleft   = ind_downleft.size();
  n_send_downright  = ind_downright.size();

  // Calculate forces within the cell first - intra cell

  for (int i=0; i<num_atoms; i++)     // loop over atoms within the cell..
  {
    
    for (int j=i+1; j<num_atoms; j++) {
      r = sep(atoms[i], atoms[j]);

      // if within range then add forces:
      if (r < R) {
        
        forces_temp = derivatives_LJ(r,atoms[i],atoms[j]); // get PE between atoms i,j and x,y forces on atom a
        temp_potential += forces_temp[0];
        cell_forces[0][i] += forces_temp[1];
        cell_forces[1][i] += forces_temp[2];
        cell_forces[0][j] -= forces_temp[1];       //negative of x force on a
        cell_forces[1][j] -= forces_temp[2];       //negative of y force on a
        
      }
    }
    
  }
  
  // now send and receive skins
  // Start with horizontal staggering to transfer all skins except up/down
  // Tags for sending:
  //                    SEND                                                       RECEIVE
  //              | 7 |  8   | 1 |                                            | 3 |  4   | 5 |
  //              | 6 |[cell]| 2 |                                            | 2 |[cell]| 6 |
  //              | 5 |  4   | 3 |                                            | 1 |  8   | 7 |


  //--------------------------------------Horizontal--------------------------------------
  
  // first send the number of atoms that will be sent 

  if (col%2==0) {   // send from even columns
    // send&recv right
    MPI_Ssend(&n_send_right,     1, MPI_INT, row*sqnproc+modulo(col+1,sqnproc), 2, comm);
    MPI_Recv(&n_recv_left,       1, MPI_INT, row*sqnproc+modulo(col-1,sqnproc), 2, comm, MPI_STATUS_IGNORE);
    // send&recv left
    MPI_Ssend(&n_send_left,      1, MPI_INT, row*sqnproc+modulo(col-1,sqnproc), 6, comm);
    MPI_Recv(&n_recv_right,      1, MPI_INT, row*sqnproc+modulo(col+1,sqnproc), 6, comm, MPI_STATUS_IGNORE);
    //Up and right
    MPI_Ssend(&n_send_upright,   1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+modulo(col+1,sqnproc), 1, comm);
    MPI_Recv(&n_recv_downleft,   1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+modulo(col-1,sqnproc), 1, comm, MPI_STATUS_IGNORE);
    //Up and left
    MPI_Ssend(&n_send_upleft,    1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+modulo(col-1,sqnproc), 7, comm);
    MPI_Recv(&n_recv_downright,  1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+modulo(col+1,sqnproc), 7, comm, MPI_STATUS_IGNORE);
    // Down and right
    MPI_Ssend(&n_send_downright, 1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+modulo(col+1,sqnproc), 3, comm);
    MPI_Recv(&n_recv_upleft,     1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+modulo(col-1,sqnproc), 3, comm, MPI_STATUS_IGNORE);
    // Down and left
    MPI_Ssend(&n_send_downleft,  1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+modulo(col-1,sqnproc), 5, comm);
    MPI_Recv(&n_recv_upright,    1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+modulo(col+1,sqnproc), 5, comm, MPI_STATUS_IGNORE);
    

  } else {  // send from odd columns
    // recv&send right
    MPI_Recv(&n_recv_left, 1, MPI_INT, row*sqnproc+modulo(col-1,sqnproc), 2, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&n_send_right, 1, MPI_INT, row*sqnproc+modulo(col+1,sqnproc), 2, comm);
    // recv&send left
    MPI_Recv(&n_recv_right, 1, MPI_INT, row*sqnproc+modulo(col+1,sqnproc), 6, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&n_send_left, 1, MPI_INT, row*sqnproc+modulo(col-1,sqnproc), 6, comm);
    //Up and right
    MPI_Recv(&n_recv_downleft,   1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+modulo(col-1,sqnproc), 1, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&n_send_upright,   1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+modulo(col+1,sqnproc), 1, comm);
    //Up and left
    MPI_Recv(&n_recv_downright,  1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+modulo(col+1,sqnproc), 7, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&n_send_upleft,    1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+modulo(col-1,sqnproc), 7, comm);
    // Down and right
    MPI_Recv(&n_recv_upleft,     1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+modulo(col-1,sqnproc), 3, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&n_send_downright, 1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+modulo(col+1,sqnproc), 3, comm);
    // Down and left
    MPI_Recv(&n_recv_upright,    1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+modulo(col+1,sqnproc), 5, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&n_send_downleft,  1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+modulo(col-1,sqnproc), 5, comm);

  }
  // now send the atoms that are in the skins in neighbouring cells
  
  if (col%2==0) {   // send right from even columns first then from odd columns. Then send left
    // Send right, receive left
    for (int i=0; i < n_send_right; i++) {
      MPI_Ssend(&(atoms[ind_right[i]]), 1, MPI_atom, row*sqnproc+modulo(col+1,sqnproc), i, comm);
    }
    for (int i=0; i < n_recv_left; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, row*sqnproc+modulo(col-1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      recv_left.push_back(temp);
    }
    // Send left, receive right
    for (int i=0; i < n_send_left; i++) {
      MPI_Ssend(&(atoms[ind_left[i]]), 1, MPI_atom, row*sqnproc+modulo(col-1,sqnproc), i, comm);
    }
    for (int i=0; i < n_recv_right; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, row*sqnproc+modulo(col+1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      recv_right.push_back(temp);
    }
    // Send up and right, receive down and left
    for (int i=0; i < n_send_upright; i++) {
      MPI_Ssend(&(atoms[ind_upright[i]]), 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+modulo(col+1,sqnproc), i, comm);
    }
    for (int i=0; i < n_recv_downleft; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+modulo(col-1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      recv_downleft.push_back(temp);
    }
    // Send up and left, receive down and right
    for (int i=0; i < n_send_upleft; i++) {
      MPI_Ssend(&(atoms[ind_upleft[i]]), 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+modulo(col-1,sqnproc), i, comm);
    }
    for (int i=0; i < n_recv_downright; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+modulo(col+1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      recv_downright.push_back(temp);
    }
    // Send down and right, receive up and left
    for (int i=0; i < n_send_downright; i++) {
      MPI_Ssend(&(atoms[ind_downright[i]]), 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+modulo(col+1,sqnproc), i, comm);
    }
    for (int i=0; i < n_recv_upleft; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+modulo(col-1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      recv_upleft.push_back(temp);
    }
    // Send down and left, receive up and right
    for (int i=0; i < n_send_downleft; i++) {
      MPI_Ssend(&(atoms[ind_downleft[i]]), 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+modulo(col-1,sqnproc), i, comm);
    }
    for (int i=0; i < n_recv_upright; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+modulo(col+1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      recv_upright.push_back(temp);
    }

    

  } else { // repeat for odd columns
    // Receive left, send right
    for (int i=0; i < n_recv_left; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, row*sqnproc+modulo(col-1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      recv_left.push_back(temp);
    }
    for (int i=0; i < n_send_right; i++) {
      MPI_Ssend(&(atoms[ind_right[i]]), 1, MPI_atom, row*sqnproc+modulo(col+1,sqnproc), i, comm);
    }
    // Receive right, send left
    for (int i=0; i < n_recv_right; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, row*sqnproc+modulo(col+1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      recv_right.push_back(temp);
    }
    for (int i=0; i < n_send_left; i++) {
      MPI_Ssend(&(atoms[ind_left[i]]), 1, MPI_atom, row*sqnproc+modulo(col-1,sqnproc), i, comm);
    }
    // Receive down and left, send up and right
   for (int i=0; i < n_recv_downleft; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+modulo(col-1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      recv_downleft.push_back(temp);
    }
    for (int i=0; i < n_send_upright; i++) {
      MPI_Ssend(&(atoms[ind_upright[i]]), 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+modulo(col+1,sqnproc), i, comm);
    }
     // Receive down and right, send up and left
   for (int i=0; i < n_recv_downright; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+modulo(col+1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      recv_downright.push_back(temp);
    }
    for (int i=0; i < n_send_upleft; i++) {
      MPI_Ssend(&(atoms[ind_upleft[i]]), 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+modulo(col-1,sqnproc), i, comm);
    }
     // Receive up and left, send down and right
   for (int i=0; i < n_recv_upleft; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+modulo(col-1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      recv_upleft.push_back(temp);
    }
    for (int i=0; i < n_send_downright; i++) {
      MPI_Ssend(&(atoms[ind_downright[i]]), 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+modulo(col+1,sqnproc), i, comm);
    }
     // Receive up and right, send down and left
   for (int i=0; i < n_recv_upright; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+modulo(col+1,sqnproc), i, comm, MPI_STATUS_IGNORE);
      recv_upright.push_back(temp);
    }
    for (int i=0; i < n_send_downleft; i++) {
      MPI_Ssend(&(atoms[ind_downleft[i]]), 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+modulo(col-1,sqnproc), i, comm);
    }

  }
  //---------------------- Send y direction - same col, row+-1 ---------------------------------------------
  
  // now communicate the number of atoms that will be sent across horizontal borders
  
  if (row%2==0) {   // send up from even row first then from odd rows. Then send down
    
    MPI_Ssend(&n_send_up, 1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+col, 8, comm);
    MPI_Recv(&n_recv_down, 1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+col, 8, comm, MPI_STATUS_IGNORE);
    
    MPI_Ssend(&n_send_down, 1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+col, 4, comm);
    MPI_Recv(&n_recv_up, 1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+col, 4, comm, MPI_STATUS_IGNORE);
    

  } else {
    
    MPI_Recv(&n_recv_down, 1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+col, 8, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&n_send_up, 1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+col, 8, comm);
    
    MPI_Recv(&n_recv_up, 1, MPI_INT, modulo(row+1,sqnproc)*sqnproc+col, 4, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&n_send_down, 1, MPI_INT, modulo(row-1,sqnproc)*sqnproc+col, 4, comm);
  }

  // now actually sending the atoms that crossed process boundaries - do this first due to following if statements
  
  if (row%2==0) {   // send up from even rows first then from odd rows. Then send down
    
    for (int i=0; i < n_send_up; i++) {
      MPI_Ssend(&(atoms[ind_up[i]]), 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+col, i, comm);
    }
    for (int i=0; i < n_recv_down; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+col, i, comm, MPI_STATUS_IGNORE);
      recv_down.push_back(temp);
    }
    
    for (int i=0; i < n_send_down; i++) {
      MPI_Ssend(&(atoms[ind_down[i]]), 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+col, i, comm);
    }
    for (int i=0; i < n_recv_up; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+col, i, comm, MPI_STATUS_IGNORE);
      recv_up.push_back(temp);
    }
    

  } else {
    
    for (int i=0; i < n_recv_down; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+col, i, comm, MPI_STATUS_IGNORE);
      recv_down.push_back(temp);
    }
    for (int i=0; i < n_send_up; i++) {
      MPI_Ssend(&(atoms[ind_up[i]]), 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+col, i, comm);
    }
    
    for (int i=0; i < n_recv_up; i++) {
      Atom temp;
      MPI_Recv(&temp, 1, MPI_atom, modulo(row+1,sqnproc)*sqnproc+col, i, comm, MPI_STATUS_IGNORE);
      recv_up.push_back(temp);
    }
    for (int i=0; i < n_send_down; i++) {
      MPI_Ssend(&(atoms[ind_down[i]]), 1, MPI_atom, modulo(row-1,sqnproc)*sqnproc+col, i, comm);
    }
    
  }
  // now edit received atoms for processes on border to account for wrapping around the boundary
  
  if (col == 0) {
    for (int i=0; i < n_recv_left; i++) {
      recv_left[i].x[0] -= L;
    }
    for (int i=0; i<n_recv_upleft; i++)
    {
        recv_upleft[i].x[0] -= L;
    }
    for (int i=0; i<n_recv_downleft; i++)
    {
        recv_downleft[i].x[0] -= L;
    }

  }
  if (col == sqnproc-1) {
    for (int i=0; i < n_recv_right; i++) {
      recv_right[i].x[0] += L;
    }
    for (int i=0; i<n_recv_upright; i++)
    {
        recv_upright[i].x[0] += L;
    }
    for (int i=0; i<n_recv_downright; i++)
    {
        recv_downright[i].x[0] += L;
    }
  }
  if (row == 0) {
    for (int i=0; i < n_recv_down; i++) {
      recv_down[i].x[1] -= L;
    }
    for (int i=0; i<n_recv_downright; i++)
    {
        recv_downright[i].x[1] -= L;
    }
    for (int i=0; i<n_recv_downleft; i++)
    {
        recv_downleft[i].x[1] -= L;
    }
  }
  if (row == sqnproc-1) {
    for (int i=0; i < n_recv_up; i++) {
      recv_up[i].x[1] += L;
    }
    for (int i=0; i<n_recv_upright; i++)
    {
        recv_upright[i].x[1] += L;
    }
    for (int i=0; i<n_recv_upleft; i++)
    {
        recv_upleft[i].x[1] += L;
    }
  }
  
  
  // now loop over corresponding skins
  
  skin_forces_indices_atoms(ind_up, atoms, recv_up, cell_forces, temp_potential);
  skin_forces_indices_atoms(ind_upright, atoms, recv_upright, cell_forces, temp_potential);
  skin_forces_indices_atoms(ind_right, atoms, recv_right, cell_forces, temp_potential);
  skin_forces_indices_atoms(ind_downright, atoms, recv_downright, cell_forces, temp_potential);
  skin_forces_indices_atoms(ind_down, atoms, recv_down, cell_forces, temp_potential);
  skin_forces_indices_atoms(ind_downleft, atoms, recv_downleft, cell_forces, temp_potential);
  skin_forces_indices_atoms(ind_left, atoms, recv_left, cell_forces, temp_potential);
  skin_forces_indices_atoms(ind_upleft, atoms, recv_upleft, cell_forces, temp_potential);
  
  cell_energy.push_back(temp_potential);

  return cell_forces;
}


// implements symplectic verlet stepping scheme as an integrator
void verlet(std::vector<Atom>& atoms,std::vector< std::vector<double> >& forces_step, std::vector<double> &potential_energy, std::vector<double> &kinetic_energy, double h, double L, double R, double M, int sqnproc, std::string boundary_type="periodic") 
{

    // half step velocity with exisiting forces
    for (int i = 0; i < atoms.size(); i++)
    {
    atoms[i].v[0] += forces_step[0][i] * h / 2.0;
    atoms[i].v[1] += forces_step[1][i] * h / 2.0;
    }

    // full position step
    for (int i=0; i < atoms.size(); i++)
    {
        atoms[i].x[0] += atoms[i].v[0] * h;
        atoms[i].x[1] += atoms[i].v[1] * h;
    }        

    MPI_Barrier(comm);

    parallel_enforce_boundary(atoms, L, M, sqnproc, boundary_type);         // Boundary Conditions
    forces_step = update_force(atoms, potential_energy, R, M, sqnproc);         // calculate forces

    // second half velocity step with new forces
    for (int i=0; i < atoms.size(); i++)
    {
        atoms[i].v[0] += forces_step[0][i] * h/2.0;
        atoms[i].v[1] += forces_step[1][i] * h/2.0;
    }

    kinetic_energy.push_back(kinetic(atoms));
}

// initialise atoms into a grid in each process
void initialise_parallel(int size, double delta, std::vector<double> velocity, double M, std::vector<Atom> &atoms, int rank, int nproc)
{ // M = (sub)domain size (can be R),
  // size = number of atoms per side of the domain
  // delta = distance between atoms
  // atoms = vector of atoms
  // velocity = vector for velocities corr. to atoms in atoms vector
  double start_d = 0;
  double x_top = 0, x_bot = 0, y_top = 0, y_bot = 0;
  

  if (size % 2 == 0)
  {
    start_d = 0.5 * delta + (size / 2 - 1) * delta;
  }
  else
  {
    start_d = size * delta;
  }

  for (int i = 0; i < size / 2; i++)
  {
    x_top = col * M + M / 2 - start_d;
    x_bot = col * M + M / 2 + start_d;

    y_top = row * M + M / 2 + start_d - i * delta;
    y_bot = row * M + M / 2 - start_d + i * delta;

    for (int j = 0; j < size; j++)
    {
      atoms.push_back(InitialiseAtom(x_top + j * delta, y_top, velocity[0], velocity[1]));
      atoms.push_back(InitialiseAtom(x_bot - j * delta, y_bot, velocity[0], velocity[1]));
    }
  }
  if (size % 2 != 0)
  {
    for (int j = 0; j < size; j++)
    {
      atoms.push_back(InitialiseAtom(x_top + j * delta, M / 2, velocity[0], velocity[1]));
    }
  }
}



int main()
{
  // --------- Timing variables and Initialising MPI----------------------
  
  double t_start, t_end, run_time;
  double t_start_print, t_end_print, run_time_print;
  MPI_Init(NULL,NULL);         //initialise MPI 

  comm  = MPI_COMM_WORLD;      //store the global communicator in comm variable
  MPI_Comm_rank(comm, &rank);  //process identifier stored in rank variable
  MPI_Comm_size(comm, &nproc);  //number of processes stored in size variable

  MPI_Barrier(comm);
  t_start = MPI_Wtime();
  
  // Create MPI_Datatype for our Atom structure
  MPI_Type_contiguous(4, MPI_DOUBLE, &MPI_atom); 
  MPI_Type_commit(&MPI_atom);
  
  // -------------- Initialising Parameters----------------
  
  // physical parameters

  int sqnproc = sqrt(nproc);            // Square root of the number of process i.e. number of cells per axis of box
  int cell_size = 16;                   // Number of atoms per side of a cell
  int Npproc = pow(cell_size, 2);       // Number of atoms per process, i.e. in 1 cell
  int Ntot = Npproc * nproc;            // Total number of atoms in the lattice
  double delta_init = pow(2, 1.0 / 6.0) * sigma;       // initial atom separation
  double M = sqrt(Npproc) * delta_init; // Length of each cell within the domain, should be greater than 2*R (cell skin on both sides)
  R = 2.5 * sigma;                      // *Cut-off radius for atom interaction*
  L = sqnproc * M;                      // Length of the entire domain
  std::vector<double> init_velocity(2, 0);  // initial velocity to use for the atoms (set to 0)

  // numerical parameters 

  double t = 0.0; double tf = 50;       // start and finish time
  int n = 500000;                       // number of verlet steps
  double step = (tf-t)/double(n);       // time step size

  std::string boundary_type="periodic"; // Type of boundary used (periodic or reflective)
  std::string write_pos="True";         // option to turn off writing positions at each step (True or False)
  
  // print out parameters to make sure they are somewhat physical
  if (rank == 0)
  {
    cout << "Total number of atoms = " << Ntot << std::endl;
    cout << "Number of atoms per process = " << Npproc << std::endl;
    cout << "Number of processes = " << nproc << std::endl;
    cout << "delta_init = " << delta_init << std::endl;
    cout << "L = " << L << std::endl;
    cout << "M = " << M << std::endl;
    cout << "R = " << R << std::endl;
  }

  //------------------- Interpolation Coefficients -----------------------
  
  a = (6.0 * pow(sigma,6.0) * ((9.0*eps - 2.0*R) * pow(eps - R,6.0) + 2.0 * (-15.0*eps + 2.0*R)* pow(sigma,6.0)))/(pow(eps,3.0)*pow(eps - R,14.0));
  
  b = (12.0*pow(sigma,6.0) * (pow(eps - R,6.0) * (5.0 * pow(eps,2.0) - 15.0*eps*R + 3.0*pow(R,2.0)) - 2.0 * (8*pow(eps,2.0) - 24*eps*R + 3.0*pow(R,2.0)) * pow(sigma,6.0)))/(pow(eps,3.0)*pow(eps - R,14.0));
  
  c = 6.0*R*pow(sigma,6.0)*(-pow(eps - R,6.0) * (20.0 * pow(eps,2.0) - 33.0*eps*R + 6.0*pow(R,2.0)) + 2.0 * (32.0 * pow(eps,2.0) - 51.0*eps*R + 6.0*pow(R,2.0))* pow(sigma,6.0))/(pow(eps,3.0)*pow(eps - R,14.0));
  
  d = (12.0*pow(R,2.0)*pow(sigma,6.0) * (pow(eps - R,6.0)*(5.0*eps - R) + 2.0 * (-8.0*eps + R) * pow(sigma,6.0)))/(pow(eps,3.0)*pow(eps - R,13.0));

  a1 = (-2.0*pow(eps-R,6.0)*(4.0*eps-R)*pow(sigma,6.0) + 2.0*(7.0*eps-R)*pow(sigma,12.0))/(pow(eps,3.0)*pow(eps - R,13.0));
  
  b1 = (-3.0*pow(eps-R,6.0)*(3.0 * pow(eps,2.0) - 9.0*eps*R + 2.0*pow(R,2.0))*pow(sigma,6.0) + 3.0*(5.0 * pow(eps,2.0) - 15.0*eps*R + 2.0*pow(R,2.0))*pow(sigma,12.0))/(pow(eps,3.0)*pow(eps - R,13.0));
  
  c1 = 6.0*R*pow(sigma,6.0)*(pow(eps-R,6.0)*(3.0 * pow(eps,2.0) - 5.0*eps*R + pow(R,2.0)) - (5.0 * pow(eps,2.0) - 8.0*eps*R + pow(R,2.0))*pow(sigma,6.0))/(pow(eps,3.0)*pow(eps - R,13.0));
  
  d1 = pow(R,2.0)*pow(sigma,6.0)*(pow(eps-R,6.0)*(-9.0*eps+2.0*R) + (15.0*eps-2.0*R)*pow(sigma,6.0))/(pow(eps,3.0)*pow(eps - R,12.0));
  
  // -------- Initialise the lattice and force and energy vectors ----------
  
  std::vector<Atom> atoms;              // vector of atoms for each process
  
  row = floor(rank / sqnproc);   // Index of square in y direction
  col = rank % sqnproc;          // Index of square in x direction

  // initialise the lattice in each rank = split atoms into subdomains equally
  initialise_parallel(cell_size, delta_init, init_velocity, M, atoms, rank, nproc);
  
  // give all the atoms a velocity kick towards the centre
  for (int i=0; i<Npproc; i++) {
    if (atoms[i].x[0] < L/2.0) {
      atoms[i].v[0] += 0.01;
    }
    else {
      atoms[i].v[0] -= 0.01;
    }
    if (atoms[i].x[1] < L/2.0) {
      atoms[i].v[1] += 0.01;
    }
    else {
      atoms[i].v[1] -= 0.01;
    }
  }

  // open files to write outputs to
  std::ofstream myfile_pos;
  std::ofstream myfile_vel;
  std::ofstream myfile_energy;

  if (rank==0) {
    if (write_pos=="True") {
      myfile_pos.open ("pos_test.dat");
      myfile_vel.open ("vel_test.dat");
    }
    
    cout << "----------" << std::endl;
    cout << "Number of Steps: " << n << std::endl;
    cout << "----------" << std::endl;
  }

  std::vector<double> potential_energy_proc;      // variable to store potential energies in at each step
  std::vector<double> kinetic_energy_proc;        // variable to store kinetic energies in at each step
  std::vector< std::vector <double> > forces_step(2, std::vector<double>(atoms.size(), 0)); // create vector to store forces between time steps

  forces_step = update_force(atoms, potential_energy_proc, R, M, sqnproc); // inital force caluclation
  kinetic_energy_proc.push_back(kinetic(atoms));   // initial kinetic energy computation
  

  // ---------------------------- MAIN LOOP ---------------------------------------------
  for (int i = 0; i<n; i++)
  {
    if ((i%10000==0) && rank==0) {
      cout << "Iteration " << i << std::endl;
    }

    verlet(atoms, forces_step, potential_energy_proc, kinetic_energy_proc, step, L, R, M, sqnproc, boundary_type);       // Take a verlet step
    

    //  Print positions at each step to file
    if (write_pos == "True")
    {
      print(atoms, myfile_pos, myfile_vel, nproc, Ntot);
    }


    MPI_Barrier(comm);              // force a sync at each step 
  } 
  // --------------------------MAIN LOOP ENDS -------------------------------------------


  // Close the output files
  if (write_pos=="True") {
    if (rank==0)
    {
      myfile_pos.close();
      myfile_vel.close();
    }
  }
  
  MPI_Barrier(comm);
  t_end = MPI_Wtime();
  run_time = t_end-t_start;

  MPI_Barrier(comm);
  
  // --------------- Gather and reduce the energy vectors on every process ----------
  
  t_start_print = MPI_Wtime();
  std::vector<double> total_potential_energy;
  std::vector<double> total_kinetic_energy;

  // resize the total energy vector to gather total energy at each time step across processes
  if (rank==0) {
    total_potential_energy.resize(n+1);
    total_kinetic_energy.resize(n+1);
  }

  // gather and sum energy at each time step
  MPI_Reduce(&(potential_energy_proc[0]), &(total_potential_energy[0]), n+1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&(kinetic_energy_proc[0]), &(total_kinetic_energy[0]), n+1, MPI_DOUBLE, MPI_SUM, 0, comm);

  // Print energy to file
  if (rank==0) {
     myfile_energy.open ("energy.dat");

    for (int i=0; i<n; i++)
    {
      myfile_energy << total_potential_energy[i] <<" "<< total_kinetic_energy[i]<< std::endl;
    }

    myfile_energy.close();
  }
  
  MPI_Barrier(comm);
  t_end_print = MPI_Wtime();
  run_time_print = t_end_print - t_start_print;

  if (rank == 0) {
    cout << "Computation run time: " << run_time << std::endl;
    cout << "Energy printing run time: " << run_time_print << std::endl;
  }
  
  MPI_Finalize(); //need to finalise MPI
  
}               // END
