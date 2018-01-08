/*
Boundary velocity propagation from one transducer to a surface through a volume, by Thomas Watson, 2017.

Modified from code by Sebastian Maurino, 2016.
Based on Ryan Jones' 2013 VelBoundary solvers.

The first medium is that between the transducer (Tx) and the surface (Sx).
The second medium is that after the surface (Sx).

[vels_T_l, vels_R_l, vels_T_s, vels_R_s] = velsurface(k1_l, k1_s, c1_l, c1_s, rho1, Tx_center, Tx_norm, Tx_ds, vels_l, vels_s, c2_l, c2_s, rho2, Sx_center, Sx_norm [,ocl_mode, ...]);

Inputs:
k1_l:       Complex wavenumber of longitudinal waves in first medium    complex double scalar
k1_s:       Complex wavenumber of shear waves in first medium           complex double scalar
c1_l:       Speed of longitudinal waves (m/s) in first medium           double scalar
c1_s:       Speed of shear waves (m/s) in first medium                  double scalar
rho1:       Density (kg/m^3) of first medium                            double scalar
Tx_center:  Center coordinates of transducer subelements                double matrix of size [NTx, 3]
Tx_norm:    Normal vectors of transducer subelements                    double matrix of size [NTx, 3]
Tx_ds:      Area of each transducer subelement                          double column vector of size [NTx]
vels_l:     Input velocity grid of longitudonal waves                   complex double column vector of size [NTx]
vels_s:     Input velocity grid of shear waves | [] if none             complex double column vector of size [NTx] | or an empty array if no shear velocities exist
c2_l:       Speed of longitudinal waves (m/s) in second medium          double scalar
c2_s:       Speed of shear waves (m/s) in second medium                 double scalar
rho2:       Density (kg/m^3) of second medium                           double scalar
Sx_center:  Center coordinate for each surface face                     double matrix of size [NSf, 3]
Sx_norm:    Normal vector for each surface face (pointing inwards!)     double matrix of size [NSf, 3]
[ocl_mode]: Oclusion mode for testing if faces are visible              integer scalar
                0: No oclusion testing (default)
                1: Surface mesh visibility test
                2: Mask visibility test

Oclusion mode 1:
[...] = velsurface(..., ocl_mode = 1, Sx_verts, Sx_faces);
    Inputs:
    Sx_verts:   Vertices that form surface mesh                         double matrix of size [NSv, 3]
    Sx_faces:   Faces that form surface mesh                            integer matrix of size [Nsv, 3]

Oclusion mode 2:
[...] = velsurface(..., ocl_mode = 2, mask, x, y, z);
    Inputs:
    mask:       Subset of voxels in volume corresponding to medium      logical matrix of size [Ny, Nx, Nz]
    x:          Values of x-dimension of each voxel in cubic volume     double column vector od size [Nx]
    y:          Values of y-dimension of each voxel in cubic volume     double column vector od size [Ny]
    z:          Values of z-dimension of each voxel in cubic volume     double column vector od size [Nz]

*/

#include <algorithm> // Max
#include <thrust/complex.h> // Complex numbers that work perfectly on device functions
#include <glm/vec3.hpp> // 3d vectors (glm::tvec3<T>)
#include <glm/trigonometric.hpp> // Trigonometric functions (glm::sin(), etc.)
#include <glm/geometric.hpp> // Geometric functions (glm::dot(), etc.)
#include <glm/gtc/constants.hpp> // General math constants (e.g. glm::pi())
#include "cudaatomicaddcomplex.hpp" // Needed to write to same address from multiple threads
#include "mex.h"
#include "matrix.h"
#include "isvisible.hpp" // Contains oclusion test functions

using namespace std ;


// Constant for threads per block to be used by CUDA. 256 is the standard
const int THREADS_PER_BLOCK = 256 ;

// Union used to send variable oclusion parameters sets (structs) into functions
union Ocl_Args
{
    struct
    {
        glm::tvec3<double>* Sx_verts ;
        glm::tvec3<int>* Sx_faces ;
        int NSv ;
    } mode1 ;

    struct
    {
        bool* mask;
        int Nx ;
        int Ny ;
        int Nz ;
        double* x ;
        double* y ;
        double* z ;
    } mode2 ;
};

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








// Solver for fluid-fluid interfaces (TMW) ------------------------------------------------------------------------------------------------------------------------------------------------------------

__device__ void velocity_Sx_Tx_fluidfluid(double c1_l, double rho1, thrust::complex<double> vels_l, double c2_l, double rho2, double thetacrit_ll, double theta_i, thrust::complex<double> *vels_T_l,
                                          thrust::complex<double> *vels_R_l)
{
    // Initialise variables
    double Z1 ;
    thrust::complex<double> theta_t, Z2, R, dv ;
    
    // Calculate transmitted angle using Snell's law (thrust outputs wrong number for > 1 values - confirmed by TMW)
    theta_t = thrust::conj(thrust::asin(thrust::complex<double>((c2_l/c1_l)*glm::sin(theta_i)))) ;

    // Compute the impedance coefficients as defined by Cobbold
    Z1 = rho1*c1_l/glm::cos(theta_i) ;
    Z2 = rho2*c2_l/thrust::cos(theta_t) ;


    // Calculate the reflection coefficient
    R = (Z2 - Z1)/(Z2 + Z1) ;

    // Calculate the reflected velocity contribution
    dv = vels_l*R*glm::cos(theta_i) ;

    // Add the reflected velocity contribution
    atomicAdd(vels_R_l, dv) ;


    // If theta_i is less than the critical angle, compute the transmission velocity contribution
    if (theta_i < thetacrit_ll)
    {
        // Calculate the transmission coefficient
        thrust::complex<double> T = (2.0*(glm::cos(theta_i)/thrust::cos(theta_t))*Z1)/(Z2 + Z1) ;

        // Calculate the transmitted velocity contribution
        dv = vels_l*T*glm::cos(theta_t.real()) ;
        
        // Add the transmitted velocity contribution
        atomicAdd(vels_T_l, dv) ;
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








// Solver for fluid-solid interfaces (TMW) ------------------------------------------------------------------------------------------------------------------------------------------------------------

__device__ void velocity_Sx_Tx_fluidsolid(double c1_l, double rho1, thrust::complex<double> vels_l, double c2_l, double c2_s, double rho2, double thetacrit_ll, double thetacrit_ls, double theta_i,
                                          thrust::complex<double> *vels_T_l, thrust::complex<double> *vels_R_l, thrust::complex<double> *vels_T_s)
{
    // Initialise variables
    double Z1 ;
    thrust::complex<double> theta_t_l, theta_t_s, Z2_l, Z2_s, R, dv, denom ;

    // Calculate transmitted angles using Snell's law (thrust outputs wrong number for > 1 values - confirmed by TMW)
    theta_t_l = thrust::conj(thrust::asin(thrust::complex<double>((c2_l/c1_l)*glm::sin(theta_i)))) ;
    theta_t_s = thrust::conj(thrust::asin(thrust::complex<double>((c2_s/c1_l)*glm::sin(theta_i)))) ;

    // Compute the impedance coefficients (Page 61, Eq. (1.115), R. Cobbold, Foundations of Biomedical Ultrasound)
    Z1   = rho1*c1_l/ glm::cos(theta_i) ;
    Z2_l = rho2*c2_l/ thrust::cos(theta_t_l) ;
    Z2_s = rho2*c2_s/ thrust::cos(theta_t_s) ;

    // The following value exists in the denominator of all of the transmission coefficients. It's faster just to calculate it once
    denom = Z2_l*thrust::pow(thrust::cos(2.0*theta_t_s), 2.0) + Z2_s*thrust::pow(thrust::sin(2.0*theta_t_s), 2.0) + Z1 ;


    // Calculate the reflection coefficient (Page 60, Eq. (1.114a), R. Cobbold, Foundations of Biomedical Ultrasound)
    R = (Z2_l*thrust::pow(thrust::cos(2.0*theta_t_s), 2.0) + Z2_s*thrust::pow(thrust::sin(2.0*theta_t_s), 2.0) - Z1)/denom ;

    // Calculate the reflected velocity contribution
    dv = vels_l*R*glm::cos(theta_i) ;

    // Add the reflected velocity contribution
    atomicAdd(vels_R_l, dv) ;


    // If theta_i is less than the longitudinal critical angle, compute the longitudinal transmission velocity contribution
    if (theta_i < thetacrit_ll)
    {
        // Calculate the longitudinal transmission coefficient
        thrust::complex<double> T_l = ((rho1*c1_l)/(rho2*c2_l))*(2.0*Z2_l*thrust::cos(2.0*theta_t_s))/denom ;

        // Calculate the longitudinal transmitted velocity contribution
        dv = vels_l*T_l*glm::cos(theta_t_l.real()) ;

        // Add the longitudinal transmitted velocity contribution
        atomicAdd(vels_T_l, dv) ;
    }


    // If theta_i is less than the shear critical angle, compute the shear transmission velocity contribution
    if (theta_i < thetacrit_ls)
    {
        // Calculate the shear transmission coefficient
        thrust::complex<double> T_s = -((rho1*c1_l)/(rho2*c2_s))*(2.0*Z2_s*thrust::sin(2.0*theta_t_s))/denom ;

        // Calculate the shear transmitted velocity contribution
        dv = vels_l*T_s*glm::sin(theta_t_l.real()) ;

        // Add the shear transmitted velocity contribution
        atomicAdd(vels_T_s, dv) ;
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








// Solver for solid-fluid interfaces with longitudinal waves ------------------------------------------------------------------------------------------------------------------------------------------

__device__ void velocity_Sx_Tx_solidfluidl(double c1_l, double c1_s, double rho1, thrust::complex<double> vels_l, double c2_l, double rho2, double thetacrit_ll, double thetacrit_sl, double theta_i, 
                                           thrust::complex<double> *vels_T_l, thrust::complex<double> *vels_R_l, thrust::complex<double> *vels_R_s)
{
    // Initialise variables
    double Z1_ll ;
    thrust::complex<double> theta_t_ll, theta_r_ls, Z2_ll, Z1_ls, R_ll, R_ls, dv, denoml ;

    // Calculate transmitted and reflected angles using Snell's law (thrust outputs wrong number for > 1 values - confirmed by TMW)
    theta_t_ll = thrust::conj(thrust::asin(thrust::complex<double>((c2_l/c1_l)*glm::sin(theta_i)))) ;
    theta_r_ls = thrust::conj(thrust::asin(thrust::complex<double>((c1_s/c1_l)*glm::sin(theta_i)))) ;

    // Compute the impedance coefficients and useful values
    Z1_ll = rho1*c1_l/glm::cos(theta_i) ;
    Z1_ls = rho1*c1_s/thrust::cos(theta_r_ls) ;
    Z2_ll = rho2*c2_l/thrust::cos(theta_t_ll) ;

    // The following value exists in the denominator of all of the transmission coefficients. It's faster just to calculate it once
    denoml = Z2_ll + Z1_ls*thrust::pow(thrust::sin(2.0*theta_r_ls), 2.0) + Z1_ll*thrust::pow(thrust::cos(2.0*theta_r_ls), 2.0) ;


    // Calculate the longitudinal reflection coefficient
    R_ll = (Z2_ll + Z1_ls*thrust::pow(thrust::sin(2.0*theta_r_ls), 2.0) - Z1_ll*thrust::pow(thrust::cos(2.0*theta_r_ls), 2.0))/denoml ;

    // Calculate the longitudinal reflected velocity contribution
    dv = vels_l*R_ll*glm::cos(theta_i) ;

    // Add the longitudinal reflected velocity contribution
    atomicAdd(vels_R_l, dv) ;


    // Calculate the shear reflection coefficient
    R_ls = -(c1_l/c1_s)*2.0*Z1_ls*thrust::sin(2.0*theta_r_ls)*thrust::cos(2.0*theta_r_ls)/denoml ;

    // Calculate the shear reflected velocity contribution
    dv = vels_l*R_ls*glm::sin(theta_r_ls.real()) ;

    // Add the longitudinal reflected velocity contribution
    atomicAdd(vels_R_s, dv) ;


    // If theta_i is less than the critical angle, compute the longitudinal transmission velocity contribution
    if (theta_i < thetacrit_ll)
    {
        // Calculate the longitudinal transmission coefficient
        thrust::complex<double> T_ll = ((rho1*c1_l)/(rho2*c2_l))*2.0*Z2_ll*thrust::cos(2.0*theta_r_ls)/denoml ;

        // Calculate the shear reflected velocity contribution
        dv = vels_l*T_ll*glm::cos(theta_t_ll.real()) ;

        // Add the longitudinal transmitted velocity contribution
        atomicAdd(vels_T_l, dv) ;
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








// Solver for solid-fluid interfaces with shear waves -------------------------------------------------------------------------------------------------------------------------------------------------

__device__ void velocity_Sx_Tx_solidfluids(double c1_l, double c1_s, double rho1, thrust::complex<double> vels_s, double c2_l, double rho2, double thetacrit_ll, double thetacrit_sl, double theta_i,
                                           thrust::complex<double> *vels_T_l, thrust::complex<double> *vels_R_l, thrust::complex<double> *vels_R_s)
{
    // Initialise variables
    double Z1_ss ;
    thrust::complex<double> theta_t_sl, theta_r_sl, Z2_sl, Z1_sl, R_ss, R_sl, dv, denoms ;

    // Calculate transmitted and reflected angles using Snell's law (thrust outputs wrong number for > 1 values - confirmed by TMW)
    theta_t_sl = thrust::conj(thrust::asin(thrust::complex<double>((c2_l/c1_s)*glm::sin(theta_i)))) ;
    theta_r_sl = thrust::conj(thrust::asin(thrust::complex<double>((c1_l/c1_s)*glm::sin(theta_i)))) ;

    // Compute the impedance coefficients and useful values
    Z1_ss = rho1*c1_s/glm::cos(theta_i) ;
    Z1_sl = rho1*c1_l/thrust::cos(theta_r_sl) ;
    Z2_sl = rho2*c2_l/thrust::cos(theta_t_sl) ;

    // The following value exists in the denominator of all of the transmission coefficients. It's faster just to calculate it once
    denoms = Z2_sl + Z1_sl*glm::pow(glm::cos(2.0*theta_i), 2.0) + Z1_ss*glm::pow(glm::sin(2.0*theta_i), 2.0) ;


    // Calculate the shear reflection coefficient
    R_ss = -(Z2_sl + Z1_sl*glm::pow(glm::cos(2.0*theta_i), 2.0) - Z1_ss*glm::pow(glm::sin(2.0*theta_i), 2.0))/denoms ;

    // Calculate the shear reflected velocity contribution
    dv = vels_s*R_ss*glm::cos(theta_i) ;

    // Add the shear reflected velocity contribution
    atomicAdd(vels_R_s, dv) ;


    // Calculate the longitudinal reflection coefficient
    R_sl = (c1_s/c1_l)*2.0*Z1_sl*glm::sin(2.0*theta_i)*glm::cos(2.0*theta_i)/denoms ;

    // Calculate the longitudinal reflected velocity contribution
    dv = vels_s*R_sl*glm::sin(theta_r_sl.real()) ;

    // Add the longitudinal reflected velocity contribution
    atomicAdd(vels_R_l, dv) ;


    // If theta_i is less than the critical angle, compute the longitudinal transmission velocity contribution
    if (theta_i < thetacrit_sl)
    {
        // Calculate the longitudinal transmission coefficient
        thrust::complex<double> T_sl = -((rho1*c1_s)/(rho2*c2_l))*2.0*Z2_sl*glm::sin(2.0*theta_i)/denoms ;

        // Calculate the longitudinal transmitted velocity contribution
        dv = vels_s*T_sl*glm::sin(theta_t_sl.real()) ;

        // Add the longitudinal transmitted velocity contribution
        atomicAdd(vels_T_l, dv) ;
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








// Solver for solid-solid interfaces with longitudinal waves ------------------------------------------------------------------------------------------------------------------------------------------

__device__ void velocity_Sx_Tx_solidsolidl(double c1_l, double c1_s, double rho1, thrust::complex<double> vels_l, double c2_l, double c2_s, double rho2, double thetacrit_ll, double thetacrit_sl,
                                           double thetacrit_ls, double thetacrit_ss, double theta_i, thrust::complex<double> *vels_T_l, thrust::complex<double> *vels_T_s,
                                           thrust::complex<double> *vels_R_l, thrust::complex<double> *vels_R_s)
{
    // Initialise variables
    double Z1_ll ;
    thrust::complex<double> theta_t_ll, theta_t_ls, theta_r_ls, Z2_ll, Z2_ls, Z1_ls, R_ll, R_ls, dv, ldenom, lpiece1, lpiece2, lpiece3, lpiece4 ;

    // Calculate transmitted and reflected angles using Snell's law (thrust outputs wrong number for > 1 values - confirmed by TMW)
    theta_t_ll = thrust::conj(thrust::asin(thrust::complex<double>((c2_l/c1_l)*glm::sin(theta_i)))) ;
    theta_t_ls = thrust::conj(thrust::asin(thrust::complex<double>((c2_s/c1_l)*glm::sin(theta_i)))) ;
    theta_r_ls = thrust::conj(thrust::asin(thrust::complex<double>((c1_s/c1_l)*glm::sin(theta_i)))) ;

    // Compute the impedance coefficients and useful values
    Z1_ll = rho1*c1_l/glm::cos(theta_i) ;
    Z1_ls = rho1*c1_s/thrust::cos(theta_r_ls) ;
    Z2_ll = rho2*c2_l/thrust::cos(theta_t_ll) ;
    Z2_ls = rho2*c2_s/thrust::cos(theta_t_ls) ;

    // The following values exist in some form in the following transmission coefficients. It's faster just to calculate them once
    lpiece1 = thrust::cos(2.0*theta_t_ls)*(1.0 + 2.0*(rho1/rho2)*(thrust::pow(thrust::sin(theta_r_ls), 2.0)/thrust::cos(2.0*theta_t_ls))) ;
    lpiece2 = thrust::cos(2.0*theta_r_ls)*(1.0 - (rho2/rho1)*thrust::cos(2.0*theta_t_ls)/thrust::cos(2.0*theta_r_ls)) ;
    lpiece3 = thrust::sin(2.0*theta_r_ls)*(1.0 - (rho2/rho1)*(thrust::pow(thrust::sin(theta_t_ls), 2.0)/thrust::pow(thrust::sin(theta_r_ls), 2.0))) ;
    lpiece4 = thrust::cos(2.0*theta_r_ls)*(1.0 + 2.0*(rho2/rho1)*(thrust::pow(thrust::sin(theta_t_ls), 2.0)/thrust::cos(2.0*theta_r_ls))) ;
    ldenom = Z2_ll*Z2_ls*thrust::pow(lpiece1,2.0) + Z1_ll*Z1_ls*thrust::tan(theta_t_ll)*thrust::tan(theta_t_ls)*thrust::pow(lpiece2,2.0) + Z1_ls*Z1_ls*thrust::pow(lpiece3,2.0) + 
             Z1_ll*Z1_ls*thrust::pow(lpiece4,2.0) + Z2_ll*Z1_ls + Z1_ll*Z2_ls ;


    // Calculate the longitudinal reflection coefficient
    R_ll = (Z2_ll*Z2_ls*thrust::pow(lpiece1,2.0) - Z1_ll*Z1_ls*thrust::tan(theta_t_ll)*thrust::tan(theta_t_ls)*thrust::pow(lpiece2,2.0) + Z1_ls*Z1_ls*thrust::pow(lpiece3,2.0) -
            Z1_ll*Z1_ls*thrust::pow(lpiece4,2.0) + Z2_ll*Z1_ls - Z1_ll*Z2_ls)/ldenom ;

    // Calculate the longitudinal reflected velocity contribution
    dv = vels_l*R_ll*glm::cos(theta_i) ;

    // Add the longitudinal reflected velocity contribution
    atomicAdd(vels_R_l, dv) ;


    // Calculate the shear reflection coefficient
    R_ls = (c1_l/c1_s)*2.0*Z1_ls*(Z2_ll*thrust::tan(theta_t_ls)*lpiece1*lpiece2 + Z1_ls*lpiece3*lpiece4)/ldenom ;

    // Calculate the shear reflected velocity contribution
    dv = vels_l*R_ls*glm::sin(theta_r_ls.real()) ;

    // Add the shear reflected velocity contribution
    atomicAdd(vels_R_s, dv) ;


    // If theta_i is less than the longitudinal critical angle, compute the longitudinal transmission velocity contribution
    if (theta_i < thetacrit_ll)
    {
        // Calculate the longitudinal transmission coefficient
        thrust::complex<double> T_ll = ((rho1*c1_l)/(rho2*c2_l))*2.0*Z2_ll*(Z2_ls*lpiece1 + Z1_ls*lpiece4)/ldenom ;

        // Calculate the longitudinal transmitted velocity contribution
        dv = vels_l*T_ll*glm::cos(theta_t_ll.real()) ;

        // Add the longitudinal transmitted velocity contribution component-wise
        atomicAdd(vels_T_l, dv) ;
    }

    // If theta_i is less than the shear critical angle, compute the shear transmission velocity contribution
    if (theta_i < thetacrit_ls)
    {
        // Calculate the shear transmission coefficient
        thrust::complex<double> T_ls = ((rho1*c1_l)/(rho2*c2_s))*2.0*Z2_ls*Z1_ls*(lpiece3 - thrust::tan(theta_t_ll)*lpiece2)/ldenom ;

        // Calculate the shear transmitted velocity contribution
        dv = vels_l*T_ls*glm::sin(theta_t_ls.real()) ;

        // Add the shear transmitted velocity contribution component-wise
        atomicAdd(vels_T_s, dv) ;
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








// Solver for solid-solid interfaces with shear waves -------------------------------------------------------------------------------------------------------------------------------------------------

__device__ void velocity_Sx_Tx_solidsolids(double c1_l, double c1_s, double rho1, thrust::complex<double> vels_s, double c2_l, double c2_s, double rho2, double thetacrit_ll, double thetacrit_sl,
                                           double thetacrit_ls, double thetacrit_ss, double theta_i, thrust::complex<double> *vels_T_l, thrust::complex<double> *vels_T_s,
                                           thrust::complex<double> *vels_R_l, thrust::complex<double> *vels_R_s)
{
    // Initialise variables
    double Z1_ss ;
    thrust::complex<double> theta_t_sl, theta_t_ss, theta_r_sl, Z2_sl, Z2_ss, Z1_sl, R_ss, R_sl, dv, sdenom, spiece1, spiece2, spiece3, spiece4 ;

    // Calculate transmitted and reflected angles using Snell's law (thrust outputs wrong number for > 1 values - confirmed by TMW)
    theta_t_sl = thrust::conj(thrust::asin(thrust::complex<double>((c2_l/c1_s)*glm::sin(theta_i)))) ;
    theta_t_ss = thrust::conj(thrust::asin(thrust::complex<double>((c2_s/c1_s)*glm::sin(theta_i)))) ;
    theta_r_sl = thrust::conj(thrust::asin(thrust::complex<double>((c1_l/c1_s)*glm::sin(theta_i)))) ;

    // Compute the impedance coefficients and useful values
    Z1_ss = rho1*c1_s / glm::cos(theta_i);
    Z1_sl = rho1*c1_l / thrust::cos(theta_r_sl);
    Z2_ss = rho2*c2_s / thrust::cos(theta_t_ss);
    Z2_sl = rho2*c2_l / thrust::cos(theta_t_sl);

    // The following values exist in some form in the following transmission coefficients. It's faster just to calculate them once
    spiece1 = thrust::cos(2.0*theta_t_ss)*(1.0 + 2.0*(rho1/rho2)*(glm::pow(glm::sin(theta_i), 2.0)/thrust::cos(2.0*theta_t_ss)));
    spiece2 = glm::cos(2.0*theta_i)*(1.0 - (rho2/rho1)*thrust::cos(2.0*theta_t_ss)/glm::cos(2.0*theta_i));
    spiece3 = glm::sin(2.0*theta_i)*(1.0 - (rho2/rho1)*(thrust::pow(thrust::sin(theta_t_ss), 2.0)/glm::pow(glm::sin(theta_i), 2.0)));
    spiece4 = glm::cos(2.0*theta_i)*(1.0 + 2.0*(rho2/rho1)*(thrust::pow(thrust::sin(theta_t_ss), 2.0)/glm::cos(2.0*theta_i)));
    sdenom = Z2_sl*Z2_ss*thrust::pow(spiece1,2.0) + Z1_sl*Z1_ss*thrust::tan(theta_t_sl)*thrust::tan(theta_t_ss)*thrust::pow(spiece2,2.0) + Z1_ss*Z1_ss*thrust::pow(spiece3,2.0) +
    Z1_sl*Z1_ss*thrust::pow(spiece4,2.0) + Z1_sl*Z2_ss + Z2_sl*Z1_ss;


    // Calculate the shear reflection coefficient
    R_ss = (Z2_sl*Z2_ss*thrust::pow(spiece1,2.0) - Z1_sl*Z1_ss*thrust::tan(theta_t_sl)*thrust::tan(theta_t_ss)*thrust::pow(spiece2,2.0) + Z1_ss*Z1_ss*thrust::pow(spiece3,2.0) -
           Z1_sl*Z1_ss*thrust::pow(spiece4,2.0) + Z1_sl*Z2_ss - Z2_sl*Z1_ss)/sdenom ;

    // Calculate the shear reflected velocity contribution
    dv = vels_s*R_ss*glm::cos(theta_i) ;

    // Add the shear reflected velocity contribution component-wise
    atomicAdd(vels_R_s, dv) ;


    // Calculate the longitudinal reflection coefficient
    R_sl = -(c1_s/c1_l)*2.0*Z1_sl*(Z2_sl*thrust::tan(theta_t_ss)*spiece1*spiece2 + Z1_ss*spiece3*spiece4)/sdenom ;

    // Calculate the longitudinal reflected velocity contribution
    dv = vels_s*R_sl*glm::sin(theta_r_sl.real()) ;

    // Add the longitudinal reflected velocity contribution
    atomicAdd(vels_R_l, dv) ;


    // If theta_i is less than the shear critical angle, compute the shear transmission velocity contribution
    if (theta_i < thetacrit_ss)
    {
        // Calculate the shear transmission coefficient
        thrust::complex<double> T_ss = ((rho1*c1_s)/(rho2*c2_s))*2.0*Z2_ss*(Z2_sl*spiece1 + Z1_sl*spiece4)/sdenom ;

        // Calculate the shear transmitted velocity contribution
        dv = vels_s*T_ss*glm::cos(theta_t_ss.real()) ;

        // Add the shear transmitted velocity contribution
        atomicAdd(vels_T_s, dv) ;
    }

    // If theta_i is less than the longitudinal critical angle, compute the longitudinal transmission velocity contribution
    if (theta_i < thetacrit_sl)
    {
        // Calculate the longitudinal transmission coefficient
        thrust::complex<double> T_sl = -((rho1*c1_s)/(rho2*c2_l))*2.0*Z2_sl*(Z1_ss*spiece3 - Z1_sl*thrust::tan(theta_t_ss)*spiece2)/sdenom ;

        // Calculate the longitudinal transmitted velocity contribution
        dv = vels_s*T_sl*glm::sin(theta_t_sl.real()) ;

        // Add the longitudinal transmitted velocity contribution component-wise
        atomicAdd(vels_T_l, dv) ;
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------









// Main Cuda kernel function. Will calculate the velocity at each surface element in Sx due to the propagating velocity from each transducer subelement in Tx -----------------------------------------

__global__ void velocity_Sx_Tx(thrust::complex<double> k1_l, thrust::complex<double> k1_s, double c1_l, double c1_s, double rho1, glm::tvec3<double> *Tx_center, glm::tvec3<double> *Tx_norm,
                               double *Tx_ds, thrust::complex<double> *vels_l, thrust::complex<double> *vels_s, double c2_l, double c2_s, double rho2, glm::tvec3<double> *Sx_center,
                               glm::tvec3<double> *Sx_norm, int ocl_mode, Ocl_Args ocl_args, int NTx, int NSf, double thetacrit_ll, double thetacrit_ls, double thetacrit_sl, double thetacrit_ss, 
                               thrust::complex<double> *vels_T_l, thrust::complex<double> *vels_R_l, thrust::complex<double> *vels_T_s, thrust::complex<double> *vels_R_s, dim3 gridIdx)
{
    // Define the complex constant I
    const thrust::complex<double> I = {0, 1} ; //Imaginary constant

    // Get indices and exit if out of problem limits
    int Tx_i = threadIdx.x + (blockIdx.x * blockDim.x) + (gridIdx.x * gridDim.x * blockDim.x) ;
    int Sx_i = threadIdx.y + (blockIdx.y * blockDim.y) + (gridIdx.y * gridDim.y * blockDim.y) ;

    if (Tx_i >= NTx || Sx_i >= NSf)
        return ;

    // Define some key vectors
    glm::tvec3<double> ray = Sx_center[Sx_i] - Tx_center[Tx_i] ;
    glm::tvec3<double> Sx_n = Sx_norm[Sx_i] ;
    glm::tvec3<double> Tx_n = Tx_norm[Tx_i] ;

    // Find the length of r
    double r = norm3d(ray.x,ray.y,ray.z) ;

    // Normalize ray
    ray = glm::normalize(ray) ;

    // Find cosine of the angle between ray and Sx_n
    double rayS_ndotpro = glm::dot(ray,Sx_n) ; 

    // Determine if the current face is facing the current transducer subelement
    // Check the angle between the surface normals and the connecting segment
    if (r  < 0.0000001 || //is it same face
        rayS_ndotpro <= 0 || glm::dot(ray, Tx_n) <= 0)  //is it facing Tx
        return; //face is not visible

    // Perform oclusion test based on oclusion mode
    bool vis = true ;
    switch(ocl_mode)
    {
        case 1: vis = isVisibleMesh(Tx_center[Tx_i], Sx_i, ocl_args.mode1.Sx_verts, ocl_args.mode1.Sx_faces, Sx_center, NSf) ;
                break ;
        case 2: vis = isVisibleMask(Tx_center[Tx_i], Sx_center[Sx_i], ocl_args.mode2.mask, ocl_args.mode2.Nx, ocl_args.mode2.Ny, ocl_args.mode2.Nz, ocl_args.mode2.x, ocl_args.mode2.y,
                                    ocl_args.mode2.z) ;
                break ;
    }

    if (!vis)
        return ; //face is ocluded

    // Get the incident angle
    double theta_i = glm::acos(rayS_ndotpro) ;
    if (isnan(theta_i) || theta_i < 0.000001)
    {
        theta_i = 0.000001 ;
    }

    // Propagate longitudinal velocity from source to target
    thrust::complex<double> v_l = ((I*k1_l.real())/glm::two_pi<double>())*vels_l[Tx_i]*(thrust::exp(-I*k1_l*r)/r)*(1.0 - I/(k1_l.real()*r))*Tx_ds[Tx_i] ;


    if (c1_s <= 0 && c2_s <= 0)
    {
        velocity_Sx_Tx_fluidfluid(c1_l, rho1, v_l, c2_l, rho2, thetacrit_ll, theta_i, &vels_T_l[Sx_i], &vels_R_l[Sx_i]) ;
    }
    else if (c1_s <= 0)
    {
        velocity_Sx_Tx_fluidsolid(c1_l, rho1, v_l, c2_l, c2_s, rho2, thetacrit_ll, thetacrit_ls, theta_i, &vels_T_l[Sx_i], &vels_R_l[Sx_i], &vels_T_s[Sx_i]) ;
    }
    else if (c2_s <= 0)
    {
        velocity_Sx_Tx_solidfluidl(c1_l, c1_s, rho1, v_l, c2_l, rho2, thetacrit_ll, thetacrit_sl, theta_i, &vels_T_l[Sx_i], &vels_R_l[Sx_i], &vels_R_s[Sx_i]) ;
    }
    else
    {
        velocity_Sx_Tx_solidsolidl(c1_l, c1_s, rho1, v_l, c2_l, c2_s, rho2, thetacrit_ll, thetacrit_sl, thetacrit_ls, thetacrit_ss, theta_i, &vels_T_l[Sx_i], &vels_T_s[Sx_i], &vels_R_l[Sx_i],
                                   &vels_R_s[Sx_i]) ;
    }

    if (c1_s > 0)
    {
        // Propagate shear velocity from source to target
        thrust::complex<double> v_s = ((I*k1_s.real())/glm::two_pi<double>())*vels_s[Tx_i]*(thrust::exp(-I*k1_s*r)/r)*(1.0 - I/(k1_s.real()*r))*Tx_ds[Tx_i] ;
        if (c2_s <= 0)
        {
            velocity_Sx_Tx_solidfluids(c1_l, c1_s, rho1, v_s, c2_l, rho2, thetacrit_ll, thetacrit_sl, theta_i, &vels_T_l[Sx_i], &vels_R_l[Sx_i], &vels_R_s[Sx_i]) ;
        }
        else
        {
            velocity_Sx_Tx_solidsolids(c1_l, c1_s, rho1, v_s, c2_l, c2_s, rho2, thetacrit_ll, thetacrit_sl, thetacrit_ls, thetacrit_ss, theta_i, &vels_T_l[Sx_i], &vels_T_s[Sx_i], &vels_R_l[Sx_i],
                                       &vels_R_s[Sx_i]) ;
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








// Checks for errors and stops the program if there is one --------------------------------------------------------------------------------------------------------------------------------------------

void cudaErrorCheck(char* action)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        char buffer[300];
        sprintf (buffer, "CUDA Error occurred while %s: %s.", action, cudaGetErrorString(err));
        mexErrMsgTxt(buffer);
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------










// Main Solver: Interfaces with CUDA // ---------------------------------------------------------------------------------------------------------------------------------------------------------------

void solveVelocity(thrust::complex<double> k1_l, thrust::complex<double> k1_s, double c1_l, double c1_s, double rho1, glm::tvec3<double> *Tx_center, glm::tvec3<double> *Tx_norm, double *Tx_ds,
                   thrust::complex<double> *vels_l, thrust::complex<double> *vels_s, double c2_l, double c2_s, double rho2, glm::tvec3<double> *Sx_center, glm::tvec3<double> *Sx_norm, int ocl_mode,
                   Ocl_Args ocl_args, int NTx, int NSf, thrust::complex<double> *vels_T_l, thrust::complex<double> *vels_R_l, thrust::complex<double> *vels_T_s, thrust::complex<double> *vels_R_s)
{
    // Output GPU details to user
    int device ;
    cudaGetDevice(&device) ;
    cudaDeviceProp properties ;
    cudaGetDeviceProperties(&properties, device) ;
    mexPrintf("Running New Velocity to Surface propagator in GPU device %s\n", properties.name) ;
    mexEvalString("drawnow") ; // Forces MATLAB to output
    cudaErrorCheck("initializing device") ;


    // Allocate GPU arrays
    glm::tvec3<double> *d_Tx_center, *d_Tx_norm ;
    size_t sz_NTx_vec3double = NTx * sizeof(glm::tvec3<double>) ;
    cudaMalloc(&d_Tx_center, sz_NTx_vec3double) ;
    cudaMalloc(&d_Tx_norm, sz_NTx_vec3double) ;

    double *d_Tx_ds ;
    size_t sz_NTx_double = NTx * sizeof(double) ;
    cudaMalloc(&d_Tx_ds, sz_NTx_double) ;

    thrust::complex<double> *d_vels_l, *d_vels_s = NULL ;
    size_t sz_NTx_complexdouble = NTx * sizeof(thrust::complex<double>) ;
    cudaMalloc(&d_vels_l, sz_NTx_complexdouble) ;
    if (vels_s)
        cudaMalloc(&d_vels_s, sz_NTx_complexdouble) ;

    glm::tvec3<double> *d_Sx_center, *d_Sx_norm ;
    size_t sz_NSf_vec3double = NSf * sizeof(glm::tvec3<double>) ;
    cudaMalloc(&d_Sx_center, sz_NSf_vec3double) ;
    cudaMalloc(&d_Sx_norm, sz_NSf_vec3double) ;

    thrust::complex<double> *d_vels_T_l, *d_vels_R_l, *d_vels_T_s = NULL, *d_vels_R_s = NULL ;

    size_t sz_NSf_complexdouble = NSf * sizeof(thrust::complex<double>) ;
    cudaMalloc(&d_vels_T_l, sz_NSf_complexdouble) ;
    cudaMalloc(&d_vels_R_l, sz_NSf_complexdouble) ;
    if (vels_T_s)
        cudaMalloc(&d_vels_T_s, sz_NSf_complexdouble) ;
    if (vels_R_s)
        cudaMalloc(&d_vels_R_s, sz_NSf_complexdouble) ;

    // Allocate for occlusion modes
    Ocl_Args d_ocl_args ;
    size_t sz_NSv_vec3double, sz_NSf_vec3int ; // Mode 1
    size_t sz_voxels_bool, sz_Nx_double, sz_Ny_double, sz_Nz_double ; // Mode 2
    switch(ocl_mode)
    {
        case 1:
        {
            glm::tvec3<double> *d_Sx_verts ;
            glm::tvec3<int> *d_Sx_faces ;
            int NSv = ocl_args.mode1.NSv ;

            sz_NSv_vec3double = NSv * sizeof(glm::tvec3<double>) ;
            sz_NSf_vec3int = NSf * sizeof(glm::tvec3<int>) ;
            cudaMalloc(&d_Sx_verts, sz_NSv_vec3double) ;
            cudaMalloc(&d_Sx_faces, sz_NSf_vec3int) ;

            d_ocl_args.mode1 = { d_Sx_verts, d_Sx_faces, NSv } ;

            break ;
        }
        case 2:
        {
            int Nx = ocl_args.mode2.Nx ;
            int Ny = ocl_args.mode2.Ny ;
            int Nz = ocl_args.mode2.Nz ;

            double *d_x ;
            sz_Nx_double = Nx * sizeof(double) ;
            cudaMalloc(&d_x, sz_Nx_double) ;

            double *d_y ;
            sz_Ny_double = Ny * sizeof(double) ;
            cudaMalloc(&d_y, sz_Ny_double) ;

            double *d_z ;
            sz_Nz_double = Nz * sizeof(double) ;
            cudaMalloc(&d_z, sz_Nz_double) ;

            int voxels = Nx * Ny * Nz ;
            sz_voxels_bool = voxels * sizeof(bool) ;
            bool *d_mask ;
            cudaMalloc(&d_mask, sz_voxels_bool) ;

            d_ocl_args.mode2 = { d_mask, Nx, Ny, Nz, d_x, d_y, d_z } ;

            break;
        }
    }

    cudaErrorCheck("allocating memory") ;


    // Copy arrays to Device
    cudaMemcpy(d_Tx_center, Tx_center, sz_NTx_vec3double, cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_Tx_norm, Tx_norm, sz_NTx_vec3double, cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_Tx_ds, Tx_ds, sz_NTx_double, cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_vels_l, vels_l, sz_NTx_complexdouble, cudaMemcpyHostToDevice) ;
    if (vels_s)
        cudaMemcpy(d_vels_s, vels_s, sz_NTx_complexdouble, cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_Sx_center, Sx_center, sz_NSf_vec3double, cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_Sx_norm, Sx_norm, sz_NSf_vec3double, cudaMemcpyHostToDevice) ;
    cudaMemset(d_vels_T_l, 0, sz_NSf_complexdouble) ;
    cudaMemset(d_vels_R_l, 0, sz_NSf_complexdouble) ;
    if (vels_T_s)
        cudaMemset(d_vels_T_s, 0, sz_NSf_complexdouble) ;
    if (vels_R_s)
        cudaMemset(d_vels_R_s, 0, sz_NSf_complexdouble) ;

    if (ocl_mode == 1)
    {    // For oclusion mode 1
        cudaMemcpy(d_ocl_args.mode1.Sx_verts, ocl_args.mode1.Sx_verts, sz_NSv_vec3double, cudaMemcpyHostToDevice) ;
        cudaMemcpy(d_ocl_args.mode1.Sx_faces, ocl_args.mode1.Sx_faces, sz_NSf_vec3int, cudaMemcpyHostToDevice) ;
    }
    if (ocl_mode == 2)
    {   // For occlusion mode 2
        cudaMemcpy(d_ocl_args.mode2.mask, ocl_args.mode2.mask, sz_voxels_bool, cudaMemcpyHostToDevice) ;
        cudaMemcpy(d_ocl_args.mode2.x, ocl_args.mode2.x, sz_Nx_double, cudaMemcpyHostToDevice) ;
        cudaMemcpy(d_ocl_args.mode2.y, ocl_args.mode2.y, sz_Ny_double, cudaMemcpyHostToDevice) ;
        cudaMemcpy(d_ocl_args.mode2.z, ocl_args.mode2.z, sz_Nz_double, cudaMemcpyHostToDevice) ;
    }

    cudaErrorCheck("copying memory to device") ;


    // Calculate problem-level variables
    double thetacrit_ll = glm::half_pi<double>() ; // Critical angle for longitudinal -> longitudinal waves
    double thetacrit_ls = glm::half_pi<double>() ; // Critical angle for longitudinal -> shear waves
    double thetacrit_sl = glm::half_pi<double>() ; // Critical angle for shear -> longitudinal waves
    double thetacrit_ss = glm::half_pi<double>() ; // Critical angle for shear -> shear waves
    thrust::complex<double> TIR ; // Total internal reflection angle

    TIR = thrust::asin( thrust::complex<double>(c1_l / c2_l) ) ;
    if (TIR.imag() == 0) // Critical angle only exists if real
        thetacrit_ll = TIR.real() ;

    if (c2_s > 0)
    {
        TIR = thrust::asin( thrust::complex<double>(c1_l / c2_s) ) ;
        if (TIR.imag() == 0) // Critical angle only exists if real
            thetacrit_ls = TIR.real() ;
    }

    if (c1_s > 0)
    {
        TIR = thrust::asin( thrust::complex<double>(c1_s / c2_l) ) ;
        if (TIR.imag() == 0) // Critical angle only exists if real
            thetacrit_sl = TIR.real() ;
    }

    if (c2_s > 0 && c1_s > 0)
    {
        TIR = thrust::asin( thrust::complex<double>(c1_s / c2_s) ) ;
        if (TIR.imag() == 0) // Critical angle only exists if real
            thetacrit_ss = TIR.real() ;
    }


    // Calculate size of GPU grid, blocks and threads
    // Split into separate chunks if grid size is larger than maximum
    // x: Transducer subelement
    // y: Surface mesh face
    int THREADS_PER_BLOCK_XY = sqrt(THREADS_PER_BLOCK);
    dim3 numOfGrids(((NTx/THREADS_PER_BLOCK_XY)/properties.maxGridSize[0]) + 1, ((NSf/THREADS_PER_BLOCK_XY)/properties.maxGridSize[1]) + 1, 1);
    dim3 blocksPerGrid((std::min(NTx/THREADS_PER_BLOCK_XY + 1, properties.maxGridSize[0])), (std::min(NSf/THREADS_PER_BLOCK_XY + 1, properties.maxGridSize[1])), 1);
    dim3 threadsPerBlock(THREADS_PER_BLOCK_XY, THREADS_PER_BLOCK_XY, 1);


    // Run the Kernels in the GPU
    // Each grid chunk is run in serial to avoid surpassing the maximum grid size
    for (int gx = 0; gx < numOfGrids.x; gx++)
    {
        for (int gy = 0; gy < numOfGrids.y; gy++)
        {
            dim3 gridIdx(gx, gy, 1);

            velocity_Sx_Tx <<< blocksPerGrid, threadsPerBlock >>>
            (
                k1_l, k1_s, c1_l, c1_s, rho1, d_Tx_center, d_Tx_norm, d_Tx_ds, d_vels_l, d_vels_s, c2_l, c2_s, rho2, d_Sx_center, d_Sx_norm, ocl_mode, d_ocl_args, NTx, NSf, thetacrit_ll,
                thetacrit_ls, thetacrit_sl, thetacrit_ss, d_vels_T_l, d_vels_R_l, d_vels_T_s, d_vels_R_s, gridIdx
            ) ;

            cudaDeviceSynchronize() ;
        }
    }
    cudaErrorCheck("calling kernels") ;

   // Copy results back to Host
    cudaMemcpy(vels_T_l, d_vels_T_l, sz_NSf_complexdouble, cudaMemcpyDeviceToHost) ;
    cudaMemcpy(vels_R_l, d_vels_R_l, sz_NSf_complexdouble, cudaMemcpyDeviceToHost) ;
    if (vels_T_s)
        cudaMemcpy(vels_T_s, d_vels_T_s, sz_NSf_complexdouble, cudaMemcpyDeviceToHost) ;
    if (vels_R_s)
        cudaMemcpy(vels_R_s, d_vels_R_s, sz_NSf_complexdouble, cudaMemcpyDeviceToHost) ;

    cudaErrorCheck("copying memory to host") ;


    // Free GPU buffers
    cudaFree(d_Tx_center) ;
    cudaFree(d_Tx_norm) ;
    cudaFree(d_Tx_ds) ;
    cudaFree(d_vels_l) ;
    cudaFree(d_vels_s) ;
    cudaFree(d_Sx_center) ;
    cudaFree(d_Sx_norm) ;
    cudaFree(d_vels_T_l) ;
    cudaFree(d_vels_R_l) ;
    cudaFree(d_vels_T_s) ;
    cudaFree(d_vels_R_s) ;

    if (ocl_mode == 1)
    {
        cudaFree(d_ocl_args.mode1.Sx_verts) ; // d_Sx_verts
        cudaFree(d_ocl_args.mode1.Sx_faces) ; // d_Sx_faces
    }
    if (ocl_mode == 2)
    {
        cudaFree(d_ocl_args.mode2.mask) ; // d_mask
        cudaFree(d_ocl_args.mode2.x) ; // d_x
        cudaFree(d_ocl_args.mode2.y) ; // d_y
        cudaFree(d_ocl_args.mode2.z) ; // d_z
    }

    cudaErrorCheck("freeing memory") ;

    // Reset device
    cudaDeviceReset() ;
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








// MATLAB Interfacing and Data Validating Section -----------------------------------------------------------------------------------------------------------------------------------------------------

bool mxAreSameDimensions(const mxArray *arrayPtr1, const mxArray *arrayPtr2)
{
    // Check that they have the same number of dimensions
    int numDims1, numDims2;
    numDims1 = mxGetNumberOfDimensions(arrayPtr1);
    numDims2 = mxGetNumberOfDimensions(arrayPtr2);
    if (numDims1 != numDims2)
        return false;

    // Check that all dimension sizes are the same
    for (int i = 0; i < numDims1; i++)
        if (mxGetDimensions(arrayPtr1)[i] != mxGetDimensions(arrayPtr2)[i])
            return false;

    // Everything was the same
    return true;
}

bool mxIsColumnVector(const mxArray *arrayPtr)
{
    // Check that it only has 2 dimensions
    if (mxGetNumberOfDimensions(arrayPtr) != 2)
        return false;

    // Check that it is a column vector
    if (mxGetDimensions(arrayPtr)[1] != 1)
        return false;

    // It was a column vector
    return true;
}

double mxGetVectorLength(const mxArray *arrayPtr)
{
    // Find the maximum dimension size
    const int *dims = (int*) mxGetDimensions(arrayPtr);
    int length = std::max(dims[0], dims[1]);

    // Return the found length
    return (double)length;
}

thrust::complex<double> *mxGetComplexData(const mxArray *arrayPtr)
{
    // Allocate complex double array of input size
    int numels = mxGetNumberOfElements(arrayPtr);
    if (numels == 0)
        return NULL;
    thrust::complex<double> *cplxArray = new thrust::complex<double> [numels];

    // Copy Values
    for (int i = 0; i < numels; i++)
        cplxArray[i] = thrust::complex<double> ( mxGetPr(arrayPtr)[i], mxGetPi(arrayPtr)[i] );

    // Return pointer to complex array
    return cplxArray;
}

void mxSetComplexData(const mxArray *arrayPtr, thrust::complex<double> *cplxArray, int N)
{
    // Check input is complex
    if (!mxIsComplex(arrayPtr))
        mexErrMsgTxt("input must be complex");

    // Loop through elements and set values
    for (int i = 0; i < N; i++)
    {
        mxGetPr(arrayPtr)[i] = cplxArray[i].real();
        mxGetPi(arrayPtr)[i] = cplxArray[i].imag();
    }

}

thrust::complex<double> mxGetComplexScalar(const mxArray *arrayPtr)
{
    // Create and return the complex scalar
    thrust::complex<double> out;
    if (mxIsComplex(arrayPtr))
        out = thrust::complex<double>( *mxGetPr(arrayPtr), *mxGetPi(arrayPtr) );
    else
        out = thrust::complex<double>( mxGetScalar(arrayPtr), 0 );
    return out;
}

template <typename T>
glm::tvec3<T> *mxGetVec3Data(const mxArray *arrayPtr)
{
    // Determine if vector data is in Nx3 or 3xN; assume column format
    bool isCol = true;
    if (mxGetDimensions(arrayPtr)[1] != 3)
        isCol = false;

    // Specify the spacing between elements according to format
    int spacing = isCol ? mxGetDimensions(arrayPtr)[0] : 1;

    // Allocate array of vectors
    int length =  isCol ? mxGetDimensions(arrayPtr)[0] : mxGetDimensions(arrayPtr)[1];
    glm::tvec3<T> *vec3Array = new glm::tvec3<T> [length];

    // Copy Values
    T *tmpData = (T*) mxGetData(arrayPtr);

    for (int i = 0; i < length; i++)
        vec3Array[i] = glm::tvec3<T> ( tmpData[i], tmpData[i + spacing], tmpData[i + 2*spacing] );

    // Return pointer to vec3 array
    return vec3Array;
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








// The gateway function -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ // This function checks for correct inputs, casts them to the correct data types, and calls the main solver function

    // Check for the correct number of input and output arguments
    if (nrhs != 15 && nrhs != 16 && nrhs != 18 && nrhs != 20)
        mexErrMsgTxt("Incorrect number of input arguments.\nThe correct syntax is [vels_T_l, vels_R_l, vels_T_s, vels_R_s] = velsurface(k1_l, k1_s, c1_l, c1_s, rho1, Tx_center, Tx_norm, Tx_ds, vels_l, vels_s, c2_l, c2_s, rho2, Sx_center, Sx_norm [,ocl_mode, ...])") ;
    if (nlhs != 4)
        mexErrMsgTxt("Exactly 4 output arguments are required.\nThe correct syntax is [vels_T_l, vels_R_l, vels_T_s, vels_R_s] = velsurface(k1_l, k1_s, c1_l, c1_s, rho1, Tx_center, Tx_norm, Tx_ds, vels_l, vels_s, c2_l, c2_s, rho2, Sx_center, Sx_norm [,ocl_mode, ...])") ;


    // Make input arguments more legible
    const mxArray *k1_l_in = prhs[0], *k1_s_in = prhs[1], *c1_l_in = prhs[2], *c1_s_in = prhs[3], *rho1_in = prhs[4], *Tx_center_in = prhs[5], *Tx_norm_in = prhs[6], *Tx_ds_in = prhs[7],
                  *vels_l_in = prhs[8], *vels_s_in = prhs[9], *c2_l_in = prhs[10], *c2_s_in = prhs[11], *rho2_in = prhs[12], *Sx_center_in = prhs[13], *Sx_norm_in = prhs[14];
    const mxArray *ocl_mode_in = nrhs > 15 ? prhs[15] : NULL;
    const mxArray *Sx_verts_in = nrhs > 16 ? prhs[16] : NULL, *Sx_faces_in = nrhs > 17 ? prhs[17] : NULL; // Ocl_mode 1
    const mxArray *mask_in = nrhs > 16 ? prhs[16] : NULL, *x_in = nrhs > 17 ? prhs[17] : NULL, *y_in = nrhs > 18 ? prhs[18] : NULL, *z_in = nrhs > 19 ? prhs[19] : NULL; // Ocl_mode 2



    // Check for the correct data types and sizes
    if (!mxIsDouble(k1_l_in) || !mxIsComplex(k1_l_in) || !mxIsScalar(k1_l_in))
        mexErrMsgTxt("k1_l must be a complex double scalar");
    if (!mxIsDouble(k1_s_in) || (!mxIsComplex(k1_s_in) && mxGetScalar(k1_s_in) != 0) || !mxIsScalar(k1_s_in))
        mexErrMsgTxt("k1_s must be a complex double scalar (or be 0 for fluids)");
    if (!mxIsDouble(c1_l_in) || !mxIsScalar(c1_l_in))
        mexErrMsgTxt("c1_l must be a double scalar");
    if (!mxIsDouble(c1_s_in) || !mxIsScalar(c1_s_in))
        mexErrMsgTxt("c1_s must be a double scalar (with a value of 0 for fluids)");
    if (!mxIsDouble(rho1_in) || !mxIsScalar(rho1_in))
        mexErrMsgTxt("rho1 must be a double scalar");
    if (!mxIsDouble(Tx_center_in) || mxIsScalar(Tx_center_in) || mxGetN(Tx_center_in) != 3)
        mexErrMsgTxt("Tx_center must be a double matrix of size [NTx, 3]");
    if (!mxIsDouble(Tx_norm_in) || mxIsScalar(Tx_norm_in) || !mxAreSameDimensions(Tx_center_in, Tx_norm_in))
        mexErrMsgTxt("Tx_norm must be a double matrix of size [NTx, 3]");
    if (!mxIsDouble(Tx_ds_in) || mxIsScalar(Tx_ds_in) || !mxIsColumnVector(Tx_ds_in) || mxGetVectorLength(Tx_center_in) != mxGetVectorLength(Tx_ds_in))
        mexErrMsgTxt("Tx_ds must be a double column vector of size [NTx]");
    if (!mxIsDouble(vels_l_in) || !mxIsComplex(vels_l_in) || mxIsScalar(vels_l_in) || !mxIsColumnVector(vels_l_in) || mxGetVectorLength(Tx_center_in) != mxGetVectorLength(vels_l_in))
        mexErrMsgTxt("vels_l must be a complex double column vector of size [NTx]");
    if ((!mxIsDouble(vels_s_in) || !mxIsComplex(vels_s_in) || mxIsScalar(vels_s_in) || !mxIsColumnVector(vels_s_in) || mxGetVectorLength(Tx_center_in) != mxGetVectorLength(vels_s_in))
         && !mxIsEmpty(vels_s_in))
        mexErrMsgTxt("vels_s must be a complex double column vector of size [NTx] | or an empty array if no shear velocities exist");
    if (!mxIsDouble(c2_l_in) || !mxIsScalar(c2_l_in))
        mexErrMsgTxt("c2_l must be a double scalar");
    if (!mxIsDouble(c2_s_in) || !mxIsScalar(c2_s_in))
        mexErrMsgTxt("c2_s must be a double scalar (with a value of 0 for fluids)");
    if (!mxIsDouble(rho2_in) || !mxIsScalar(rho2_in))
        mexErrMsgTxt("rho2 must be a double scalar");
    if (!mxIsDouble(Sx_center_in) || mxIsScalar(Sx_center_in) || mxGetN(Sx_center_in) != 3)
        mexErrMsgTxt("Sx_center must be a double matrix of size [NSf, 3]");
    if (!mxIsDouble(Sx_norm_in) || mxIsScalar(Sx_norm_in) || !mxAreSameDimensions(Sx_center_in, Sx_norm_in))
        mexErrMsgTxt("Sx_norm must be a double matrix of size [NSf, 3]");

    if (ocl_mode_in && (!mxIsInt32(ocl_mode_in) || !mxIsScalar(ocl_mode_in) ) )
        mexErrMsgTxt("ocl_mode must be an integer scalar");



    if (ocl_mode_in && (int)mxGetScalar(ocl_mode_in) == 1)
    {
        if (nrhs != 18)
            mexErrMsgTxt("18 input arguments are required for Oclusion Mode 1:\n[...] = velsurface(k1_l, k1_s, c1_l, c1_s, rho1, Tx_center, Tx_norm, Tx_ds, vels_l, vels_s, c2_l, c2_s, rho2, Sx_center, Sx_norm , ocl_mode=1, Sx_verts, Sx_faces)") ;
        if (!mxIsDouble(Sx_verts_in) || mxIsScalar(Sx_verts_in) || mxGetN(Sx_verts_in) != 3)
            mexErrMsgTxt("Sx_verts must be a double matrix of size [NSv, 3]") ;
        if (!mxIsInt32(Sx_faces_in) || mxIsScalar(Sx_faces_in) || !mxAreSameDimensions(Sx_center_in, Sx_faces_in))
            mexErrMsgTxt("Sx_faces must be a integer matrix of size [NSf, 3]") ;
    }

    if (ocl_mode_in && (int)mxGetScalar(ocl_mode_in) == 2)
    {
        if (nrhs != 20)
            mexErrMsgTxt("20 input arguments are required for Occlusion Mode 2:\n[...] = velsurface(k1_l, k1_s, c1_l, c1_s, rho1, Tx_center, Tx_norm, Tx_ds, vels_l, vels_s, c2_l, c2_s, rho2, Sx_center, Sx_norm , ocl_mode=2, mask, x, y, z") ;
        if (!mxIsDouble(x_in) || mxIsScalar(x_in) || !mxIsColumnVector(x_in))
            mexErrMsgTxt("x must be a double column vector") ;
        if (!mxIsDouble(y_in) || mxIsScalar(y_in) || !mxIsColumnVector(y_in))
            mexErrMsgTxt("y must be a double column vector") ;
        if (!mxIsDouble(z_in) || mxIsScalar(z_in) || !mxIsColumnVector(z_in))
            mexErrMsgTxt("z must be a double column vector") ;
        if (!mxIsLogical(mask_in) || mxIsScalar(mask_in) || mxGetNumberOfDimensions(mask_in) != 3 || mxGetDimensions(mask_in)[0] != mxGetVectorLength(y_in) || mxGetDimensions(mask_in)[1]
            != mxGetVectorLength(x_in) || mxGetDimensions(mask_in)[2] != mxGetVectorLength(z_in))
            mexErrMsgTxt("mask must be a logical matrix of size [Ny, Nx, Nz]") ;
    }


    // Get problem size
    int NTx = mxGetVectorLength(Tx_center_in) ;
    int NSf = mxGetVectorLength(Sx_center_in) ;


    // Load all data
    thrust::complex<double> k1_l, k1_s ;
    double c1_l, c1_s, rho1, c2_l, c2_s, rho2 ;
    double *Tx_ds, *x, *y, *z ;
    bool *mask ;
    glm::tvec3<double> *Tx_center, *Tx_norm, *Sx_center, *Sx_norm ;
    thrust::complex<double> *vels_l, *vels_s ;
    int ocl_mode = 0 ;

    k1_l = mxGetComplexScalar(k1_l_in) ;
    k1_s = mxGetComplexScalar(k1_s_in) ;
    c1_l = mxGetScalar(c1_l_in) ;
    c1_s = mxGetScalar(c1_s_in) ;
    rho1 = mxGetScalar(rho1_in) ;
    Tx_center = mxGetVec3Data<double>(Tx_center_in) ;
    Tx_norm = mxGetVec3Data<double>(Tx_norm_in) ;
    Tx_ds = (double*) mxGetData(Tx_ds_in) ;
    vels_l = mxGetComplexData(vels_l_in) ;
    vels_s = mxGetComplexData(vels_s_in) ;
    c2_l = mxGetScalar(c2_l_in) ;
    c2_s = mxGetScalar(c2_s_in) ;
    rho2 = mxGetScalar(rho2_in) ;
    Sx_center = mxGetVec3Data<double>(Sx_center_in) ;
    Sx_norm = mxGetVec3Data<double>(Sx_norm_in) ;
    if (ocl_mode_in) ocl_mode = (int)mxGetScalar(ocl_mode_in) ;


    // Load data for occlusion modes into union object
    Ocl_Args ocl_args ;
    switch(ocl_mode)
    {
        case 0:
            break;

        case 1:
        {
            glm::tvec3<double> *Sx_verts = mxGetVec3Data<double>(Sx_verts_in) ;
            glm::tvec3<int> *Sx_faces = mxGetVec3Data<int>(Sx_faces_in) ;

            int NSv = mxGetVectorLength(Sx_verts_in) ;

            ocl_args.mode1 = { Sx_verts, Sx_faces, NSv } ;
            break ;
        }
        case 2:
        {
            int Nx = mxGetVectorLength(x_in) ;
            int Ny = mxGetVectorLength(y_in) ;
            int Nz = mxGetVectorLength(z_in) ;

            x = (double*) mxGetData(x_in) ;
            y = (double*) mxGetData(y_in) ;
            z = (double*) mxGetData(z_in) ;

            mask = mxGetLogicals(mask_in) ;

            ocl_args.mode2 = { mask, Nx, Ny, Nz, x, y, z } ;
            break ;
        }

        default:
            mexErrMsgTxt("Invalid oclusion mode requested") ;
    }


    // Allocate necessary output arrays
    thrust::complex<double> *vels_T_l = new thrust::complex<double> [NSf] ;
    thrust::complex<double> *vels_R_l = new thrust::complex<double> [NSf] ;
    thrust::complex<double> *vels_T_s = NULL ;
    if (c2_s > 0)
        vels_T_s = new thrust::complex<double> [NSf] ;
    thrust::complex<double> *vels_R_s = NULL ;
    if (c1_s > 0)
        vels_R_s = new thrust::complex<double> [NSf] ;


    // Call main solver
    solveVelocity(k1_l, k1_s, c1_l, c1_s, rho1, Tx_center, Tx_norm, Tx_ds, vels_l, vels_s, c2_l, c2_s, rho2, Sx_center, Sx_norm, ocl_mode, ocl_args, NTx, NSf, vels_T_l, vels_R_l, vels_T_s, vels_R_s) ;


    // Create MATLAB output arrays and copy results
    plhs[0] = mxCreateDoubleMatrix(NSf, 1, mxCOMPLEX) ;
    mxSetComplexData(plhs[0], vels_T_l, NSf) ;
    plhs[1] = mxCreateDoubleMatrix(NSf, 1, mxCOMPLEX) ;
    mxSetComplexData(plhs[1], vels_R_l, NSf) ;
    if (vels_T_s)
    {
        plhs[2] = mxCreateDoubleMatrix(NSf, 1, mxCOMPLEX) ;
        mxSetComplexData(plhs[2], vels_T_s, NSf) ;
    }
    else
    {
        plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL) ;
    }

    if (vels_R_s)
    {
        plhs[3] = mxCreateDoubleMatrix(NSf, 1, mxCOMPLEX) ;
        mxSetComplexData(plhs[3], vels_R_s, NSf) ;
    }
    else
    {
        plhs[3] = mxCreateDoubleMatrix(0, 0, mxREAL) ;
    }

    delete[] vels_T_l, vels_R_l;
    if (c2_s > 0)
        delete[] vels_T_s ;
    if (c1_s > 0)
        delete[] vels_R_s ;
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------