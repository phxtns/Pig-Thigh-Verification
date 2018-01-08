/*
Penne's Bioheat Equation Solver in GPU using CUDA libraries with a MATLAB interface

by Sebastian Maurino, 2016
based on Aki Pulkkinen's 2011 Bioheat solver

Function calling syntax is:
T_end = bioheat(Domain, Qr, Qt, dh, dt, Nt, kappa, Ct, rho, Ct_b, rho_b, W_b, T_b, T0)

Inputs:
Domain: index 0 mask of all the different material types (-1 means ignore)      int32 matrix of size [Nx,Ny,Nz]
Qr:     absorbed power density distribution                                     double matrix of size [Nx,Ny,Nz]
Qt:     scalar multiple to make Qr into time dependenent: Q(t) = Qr*Qt(t)       double column vector of size Nt
dh:     spatial discretization (m)                                              double
dt:     temporal discretization (t)                                             double
Nt:     number of time steps such that sonication time = Nt*dt                  double
kappa:  thermal conductivity (W/m/K) for each of the materials in Domain        double column vector of size equal to # of materials in Domain
Ct:     specific heat (J/kg/K) for each of the materials in Domain              double column vector of size equal to # of materials in Domain
rho:    density (kg/m^3) for each of the materials in Domain                    double column vector of size equal to # of materials in Domain
Ct_b:   specific heat of blood (J/kg/K) for each of the materials in Domain     double column vector of size equal to # of materials in Domain
rho_b:  density of blood (kg/m^3) for each of the materials in Domain           double column vector of size equal to # of materials in Domain
W_b:    perfusion rate of blood (s^-1) for each of the materials in Domain      double column vector of size equal to # of materials in Domain
T_b:    temperature of blood (s^-1) for each of the materials in Domain         double column vector of size equal to # of materials in Domain
T0:     initial temperature at each spatial point                               double matrix of size [Nx,Ny,Nz]
*/

#include <algorithm>
#include "mex.h"
#include "matrix.h"

using namespace std;

//Constant for threads per block to be used by CUDA. 256 is the standard
const int THREADS_PER_BLOCK = 256;

/*----- Bioheat Solver Section -----*/


/* FDTD Function constants */
__constant__ double L0[] = { 0.0 };
__constant__ double L1[] = { -2.0, 1.0 };
__constant__ double L2[] = { -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0 };
__constant__ double L3[] = { -49.0 / 18.0, 3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0 };
__constant__ double G0[] = { 0.0 };
__constant__ double G1[] = { 0.0, 1.0 / 2.0 };
__constant__ double G2[] = { 0.0, 2.0 / 3.0, -1.0 / 12.0 };
__constant__ double G3[] = { 0.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0 };

/* ---Cuda kernel function--- */
// This function will be executed it many times in parallel in the GPU
// It uses the value of the thread ID to know which voxel to evaluate
__global__ void bioheat_voxel(int* Domain, double* Qr, double* Qt, double dh, double dt, double Nt, double* kappa, double* Ct, double* rho, double* Ct_b, double* rho_b, double* W_b, double* T_b, double* T0, double* T1, int Nx, int Ny, int Nz, int voxels, int it)
{
    /* Get voxel index */
    int ind = threadIdx.x + (blockIdx.x * blockDim.x);

    /* Consider return reasons */
    if (ind >= voxels) //only evaluate voxels in domain
        return;

    if (Domain[ind] < 0) //if Domain at this point is negative, it is indicating that the temperature change at this voxel shouldn't be evaluated (for speed)
        return;

    /* Save common computations for speed */
    const int Ny2 = 2 * Ny;
    const int Ny3 = 3 * Ny;
    const int Nxy = Nx * Ny;
    const int Nxy2 = 2 * Nx * Ny;
    const int Nxy3 = 3 * Nx * Ny;
    const double dh2 = dh*dh;

    /* Get voxel coordinates */
    int iz = ind / Nxy;
    int ix = (ind % Nxy) / Ny;
    int iy = (ind % Nxy) % Ny;

    /* Calculate maximum order according to distance to boundary*/
    int kz = min(iz, (Nz - 1) - iz); //minumum distance to z boundary
    int kx = min(ix, (Nx - 1) - ix); //minumum distance to x boundary
    int ky = min(iy, (Ny - 1) - iy); //minumum distance to y boundary
    int order = min(min(kx, ky), min(kz, 3)); //maximum possible order is 3


    /* Solve bioheat equation for this pixel and store result in T0 */
    double KappaLaPlaceT, GradKappaGradT;

    if (order == 0)
    { //Do not evaluate boundaries
        return;
    }
    else if (order >= 3)
    { //3rd order solution for this voxel
        KappaLaPlaceT = (3.0 * L3[0] * T1[ind]
            + L3[1] * (T1[ind - 1] + T1[ind + 1] + T1[ind - Ny] + T1[ind + Ny] + T1[ind - Nxy] + T1[ind + Nxy])
            + L3[2] * (T1[ind - 2] + T1[ind + 2] + T1[ind - Ny2] + T1[ind + Ny2] + T1[ind - Nxy2] + T1[ind + Nxy2])
            + L3[3] * (T1[ind - 3] + T1[ind + 3] + T1[ind - Ny3] + T1[ind + Ny3] + T1[ind - Nxy3] + T1[ind + Nxy3])) * kappa[Domain[ind]] / dh2;

        GradKappaGradT = ((G3[1] * (kappa[Domain[ind + 1]] - kappa[Domain[ind - 1]])
            + G3[2] * (kappa[Domain[ind + 2]] - kappa[Domain[ind - 2]])
            + G3[3] * (kappa[Domain[ind + 3]] - kappa[Domain[ind - 3]]))
            *
            (G3[1] * (T1[ind + 1] - T1[ind - 1])
            + G3[2] * (T1[ind + 2] - T1[ind - 2])
            + G3[3] * (T1[ind + 3] - T1[ind - 3]))
            +
            (G3[1] * (kappa[Domain[ind + Ny]] - kappa[Domain[ind - Ny]])
            + G3[2] * (kappa[Domain[ind + Ny2]] - kappa[Domain[ind - Ny2]])
            + G3[3] * (kappa[Domain[ind + Ny3]] - kappa[Domain[ind - Ny3]]))
            *
            (G3[1] * (T1[ind + Ny] - T1[ind - Ny])
            + G3[2] * (T1[ind + Ny2] - T1[ind - Ny2])
            + G3[3] * (T1[ind + Ny3] - T1[ind - Ny3]))
            +
            (G3[1] * (kappa[Domain[ind + Nxy]] - kappa[Domain[ind - Nxy]])
            + G3[2] * (kappa[Domain[ind + Nxy2]] - kappa[Domain[ind - Nxy2]])
            + G3[3] * (kappa[Domain[ind + Nxy3]] - kappa[Domain[ind - Nxy3]]))
            *
            (G3[1] * (T1[ind + Nxy] - T1[ind - Nxy])
            + G3[2] * (T1[ind + Nxy2] - T1[ind - Nxy2])
            + G3[3] * (T1[ind + Nxy3] - T1[ind - Nxy3]))) / dh2;
    }
    else if (order >= 2)
    { //2nd order solution for this voxel
        KappaLaPlaceT = (3.0 * L2[0] * T1[ind]
            + L2[1] * (T1[ind - 1] + T1[ind + 1] + T1[ind - Ny] + T1[ind + Ny] + T1[ind - Nxy] + T1[ind + Nxy])
            + L2[2] * (T1[ind - 2] + T1[ind + 2] + T1[ind - Ny2] + T1[ind + Ny2] + T1[ind - Nxy2] + T1[ind + Nxy2])) * kappa[Domain[ind]] / dh2;
        GradKappaGradT = ((G2[1] * (kappa[Domain[ind + 1]] - kappa[Domain[ind - 1]])
            + G2[2] * (kappa[Domain[ind + 2]] - kappa[Domain[ind - 2]]))
            *
            (G2[1] * (T1[ind + 1] - T1[ind - 1])
            + G2[2] * (T1[ind + 2] - T1[ind - 2]))
            +
            (G2[1] * (kappa[Domain[ind + Ny]] - kappa[Domain[ind - Ny]])
            + G2[2] * (kappa[Domain[ind + Ny2]] - kappa[Domain[ind - Ny2]]))
            *
            (G2[1] * (T1[ind + Ny] - T1[ind - Ny])
            + G2[2] * (T1[ind + Ny2] - T1[ind - Ny2]))
            +
            (G2[1] * (kappa[Domain[ind + Nxy]] - kappa[Domain[ind - Nxy]])
            + G2[2] * (kappa[Domain[ind + Nxy2]] - kappa[Domain[ind - Nxy2]]))
            *
            (G2[1] * (T1[ind + Nxy] - T1[ind - Nxy])
            + G2[2] * (T1[ind + Nxy2] - T1[ind - Nxy2]))) / dh2;
    }
    else if (order >= 1)
    { //1st order solution for this voxel
        KappaLaPlaceT = (3.0 * L1[0] * T1[ind]
            + L1[1] * (T1[ind - 1] + T1[ind + 1] + T1[ind - Ny] + T1[ind + Ny] + T1[ind - Nxy] + T1[ind + Nxy])) * kappa[Domain[ind]] / dh2;
        GradKappaGradT = ((G1[1] * (kappa[Domain[ind + 1]] - kappa[Domain[ind - 1]]))
            *
            (G1[1] * (T1[ind + 1] - T1[ind - 1]))
            +
            (G1[1] * (kappa[Domain[ind + Ny]] - kappa[Domain[ind - Ny]]))
            *
            (G1[1] * (T1[ind + Ny] - T1[ind - Ny]))
            +
            (G1[1] * (kappa[Domain[ind + Nxy]] - kappa[Domain[ind - Nxy]]))
            *
            (G1[1] * (T1[ind + Nxy] - T1[ind - Nxy]))) / dh2;
    }
    else
    { //no change in voxel values
        KappaLaPlaceT = 0.0;
        GradKappaGradT = 0.0;
    }

    //Calculate blood perfusion according to parameters at voxel
    double Perfusion = rho_b[Domain[ind]] * Ct_b[Domain[ind]] * W_b[Domain[ind]] * (T1[ind] - T_b[Domain[ind]]);

    //Calculate next T0 based on FDTD coefficients and perfusion
    T0[ind] = T1[ind] + dt * (KappaLaPlaceT + GradKappaGradT - Perfusion + Qr[ind] * Qt[it]) / (rho[Domain[ind]] * Ct[Domain[ind]]);
}

/* Checks for errors and stops the program if there is one */
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

/*---Main Bioheat Solver---*/
void solveBioheat(int* Domain, double* Qr, double* Qt, double dh, double dt, double Nt, double* kappa, double* Ct, double* rho, double* Ct_b, double* rho_b, double* W_b, double* T_b, double* T0, double* T1, int Nx, int Ny, int Nz, int materials)
{
    /* Output to user GPU details */
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    mexPrintf("Running Bioheat solver in GPU device %s\n", properties.name);

    /* Calculate maximum index of voxel */
    int voxels = Nx * Ny * Nz;

    /* Calculate size of GPU threads and blocks */
    dim3 blocksPerGrid( (voxels / THREADS_PER_BLOCK) + 1, 1, 1);
    dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);

    /* Allocate GPU arrays according to their size */
    int *d_Domain;
    size_t sz_voxels_int = voxels * sizeof(int);
    cudaMalloc(&d_Domain, sz_voxels_int);

    double *d_Qr, *d_Qt, *d_kappa, *d_Ct, *d_rho, *d_Ct_b, *d_rho_b, *d_W_b, *d_T_b, *d_T0, *d_T1;

    size_t sz_voxels_double = voxels * sizeof(double);
    cudaMalloc(&d_Qr, sz_voxels_double);
    cudaMalloc(&d_T0, sz_voxels_double);
    cudaMalloc(&d_T1, sz_voxels_double);

    size_t sz_Nt_double = Nt * sizeof(double);
    cudaMalloc(&d_Qt, sz_Nt_double);

    size_t sz_materials_double = materials * sizeof(double);
    cudaMalloc(&d_kappa, sz_materials_double);
    cudaMalloc(&d_Ct, sz_materials_double);
    cudaMalloc(&d_rho, sz_materials_double);
    cudaMalloc(&d_Ct_b, sz_materials_double);
    cudaMalloc(&d_rho_b, sz_materials_double);
    cudaMalloc(&d_W_b, sz_materials_double);
    cudaMalloc(&d_T_b, sz_materials_double);

    /* Copy input array from host to GPU */
    cudaMemcpy(d_Domain, Domain, sz_voxels_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Qr, Qr, sz_voxels_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Qt, Qt, sz_Nt_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kappa, kappa, sz_materials_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ct, Ct, sz_materials_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, rho, sz_materials_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ct_b, Ct_b, sz_materials_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho_b, rho_b, sz_materials_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_b, W_b, sz_materials_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_b, T_b, sz_materials_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T0, T0, sz_voxels_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T1, T1, sz_voxels_double, cudaMemcpyHostToDevice);

    /* Check for errors in memory allocation and copy*/
    cudaErrorCheck("allocating device memory");

    /* Declare helpful variables */
    int percent = 0;

    /* Loop through time */
    for (int it = 0; it < Nt; it++)
    {
        /* Output percentage */
        if (100 * it / Nt >= percent)
        {
            mexPrintf("%d%%\n", percent++);
            mexEvalString("drawnow"); //forces MATLAB to output
        }

        /* Run the kernels on the GPU -- one per voxel */
        bioheat_voxel <<< blocksPerGrid, threadsPerBlock >>> (d_Domain, d_Qr, d_Qt, dh, dt, Nt, d_kappa, d_Ct, d_rho, d_Ct_b, d_rho_b, d_W_b, d_T_b, d_T0, d_T1, Nx, Ny, Nz, voxels, it);

        /* Wait for all device calls to complete */
        cudaDeviceSynchronize();

        /* Check for errors in kernel calls */
        cudaErrorCheck("running CUDA kernels");

        /* Make T1 the next timestep */
        double *swap = d_T0;
        d_T0 = d_T1;
        d_T1 = swap;

    }

    /* copy the result array back to the host */
    cudaMemcpy(T1, d_T1, sz_voxels_double, cudaMemcpyDeviceToHost);
    cudaErrorCheck("copying memory to host");

    /* free GPU buffers */
    cudaFree(d_Domain);
    cudaFree(d_Qr);
    cudaFree(d_Qt);
    cudaFree(d_kappa);
    cudaFree(d_Ct);
    cudaFree(d_rho);
    cudaFree(d_Ct_b);
    cudaFree(d_rho_b);
    cudaFree(d_W_b);
    cudaFree(d_T_b);
    cudaFree(d_T0);
    cudaFree(d_T1);

    cudaErrorCheck("freeing memory");

    cudaDeviceReset();
}



/*----- MATLAB Interfacing and Data Validating Section -----*/

bool mxAreSameDimensions(const mxArray *arrayPtr1, const mxArray *arrayPtr2)
{
    /* Check that they have the same number of dimensions */
    int numDims1, numDims2;
    numDims1 = mxGetNumberOfDimensions(arrayPtr1);
    numDims2 = mxGetNumberOfDimensions(arrayPtr2);
    if (numDims1 != numDims2)
        return false;

    /* Check that all dimensions size are the same */
    for (int i = 0; i < numDims1; i++)
        if (mxGetDimensions(arrayPtr1)[i] != mxGetDimensions(arrayPtr2)[i])
            return false;

    /* Everything was the same */
    return true;
}

bool mxIsColumnVector(const mxArray *arrayPtr)
{
    /* Check that it only has 2 dimensions */
    if (mxGetNumberOfDimensions(arrayPtr) != 2)
        return false;

    /* Check that it is a column vector */
    if (mxGetDimensions(arrayPtr)[1] != 1)
        return false;

    /* It was a column vector */
    return true;
}

double mxGetVectorLength(const mxArray *arrayPtr)
{
    /*Find the maximum dimension size*/
    const int *dims = (int*) mxGetDimensions(arrayPtr);
    int length = max(dims[0], dims[1]);

    /* Return the found length */
    return (double)length;
}

/*---The gateway function---*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ //this function checks for correct inputs, casts them to the correct data types and calls the main bioheat function
    /* Check for the correct number of input and output arguments*/
    if (nrhs != 14)
        mexErrMsgTxt("Exactly 14 input arguments are required.\nThe correct syntax is T_end = bioheat(Domain, Qr, Qt, dh, dt, Nt, kappa, Ct, rho, Ct_b, rho_b, W_b, T_b, T0).\nPlease see bioheat.cpp source for more information.");
    if (nlhs != 1)
        mexErrMsgTxt("Exactly 1 output argument is required.\nThe correct syntax is T_end = bioheat(Domain, Qr, Qt, dh, dt, Nt, kappa, Ct, rho, Ct_b, rho_b, W_b, T_b, T0).\nPlease see bioheat.cpp source for more information.");

    /* Set right hand side parameter pointers to more readable names */
    const mxArray *Domain_in = prhs[0], *Qr_in = prhs[1], *Qt_in = prhs[2], *dh_in = prhs[3], *dt_in = prhs[4], *Nt_in = prhs[5], *kappa_in = prhs[6], *Ct_in = prhs[7], *rho_in = prhs[8], *Ct_b_in = prhs[9], *rho_b_in = prhs[10], *W_b_in = prhs[11], *T_b_in = prhs[12], *T0_in = prhs[13];


    /* Check for the correct data types */
    if (!mxIsInt32(Domain_in) || mxIsScalar(Domain_in))
        mexErrMsgTxt("Domain must be a int32 matrix.");
    if (!mxIsDouble(Qr_in) || mxIsScalar(Qr_in))
        mexErrMsgTxt("Qr must be a double matrix.");
    if (!mxIsDouble(Qt_in) || mxIsScalar(Qt_in) || !mxIsColumnVector(Qt_in))
        mexErrMsgTxt("Qt must be a double column vector.");
    if (!mxIsDouble(dh_in) || !mxIsScalar(dh_in))
        mexErrMsgTxt("dh must be a double scalar.");
    if (!mxIsDouble(dt_in) || !mxIsScalar(dt_in))
        mexErrMsgTxt("dt must be a double scalar.");
    if (!mxIsDouble(Nt_in) || !mxIsScalar(Nt_in))
        mexErrMsgTxt("Nt must be a double scalar.");
    if (!mxIsDouble(kappa_in) || !mxIsColumnVector(kappa_in))
        mexErrMsgTxt("kappa must be a double column vector.");
    if (!mxIsDouble(Ct_in) || !mxIsColumnVector(Ct_in))
        mexErrMsgTxt("Ct must be a double column vector.");
    if (!mxIsDouble(rho_in) || !mxIsColumnVector(rho_in))
        mexErrMsgTxt("rho must be a double column vector.");
    if (!mxIsDouble(Ct_b_in) || !mxIsColumnVector(Ct_b_in))
        mexErrMsgTxt("Ct_b must be a double column vector.");
    if (!mxIsDouble(rho_b_in) || !mxIsColumnVector(rho_b_in))
        mexErrMsgTxt("rho_b must be a double column vector.");
    if (!mxIsDouble(W_b_in) || !mxIsColumnVector(W_b_in))
        mexErrMsgTxt("W_b must be a double column vector.");
    if (!mxIsDouble(T_b_in) || !mxIsColumnVector(T_b_in))
        mexErrMsgTxt("T_b must be a double column vector.");
    if (!mxIsDouble(T0_in) || mxIsScalar(T0_in))
        mexErrMsgTxt("T0 must be a double matrix.");
    
    /* Load all data */
    int *Domain;
    double *Qr, *Qt, *kappa, *Ct, *rho, *Ct_b, *rho_b, *W_b, *T_b, *T0;
    double dh, dt, Nt;

    Domain = (int*)mxGetData(Domain_in);
    Qr = (double*)mxGetData(Qr_in);
    Qt = (double*)mxGetData(Qt_in);
    dh = mxGetScalar(dh_in);
    dt = mxGetScalar(dt_in);
    Nt = mxGetScalar(Nt_in);
    kappa = (double*)mxGetData(kappa_in);
    Ct = (double*)mxGetData(Ct_in);
    rho = (double*)mxGetData(rho_in);
    Ct_b = (double*)mxGetData(Ct_b_in);
    rho_b = (double*)mxGetData(rho_b_in);
    W_b = (double*)mxGetData(W_b_in);
    T_b = (double*)mxGetData(T_b_in);
    T0 = (double*)mxGetData(T0_in);

    /* Validate the the sizes of all matrices and vectors */
    if (mxGetNumberOfDimensions(Domain_in) != 3)
        mexErrMsgTxt("Domain must be a 3-dimensional matrix.");
    if (!mxAreSameDimensions(Domain_in, Qr_in))
        mexErrMsgTxt("Qr must be of the same size as Domain.");
    if (mxGetVectorLength(Qt_in) != Nt)
        mexErrMsgTxt("Qt must be of Nt in length.");
    int numMaterials = *max_element(Domain, Domain + mxGetNumberOfElements(Domain_in)) + 1;
    if (mxGetVectorLength(kappa_in) != numMaterials)
        mexErrMsgTxt("kappa must be of size equal to # of materials in Domain.");
    if (mxGetVectorLength(Ct_in) != numMaterials)
        mexErrMsgTxt("Ct must be of size equal to # of materials in Domain.");
    if (mxGetVectorLength(rho_in) != numMaterials)
        mexErrMsgTxt("rho must be of size equal to # of materials in Domain.");
    if (mxGetVectorLength(Ct_b_in) != numMaterials)
        mexErrMsgTxt("Ct_b must be of size equal to # of materials in Domain.");
    if (mxGetVectorLength(rho_b_in) != numMaterials)
        mexErrMsgTxt("rho_b must be of size equal to # of materials in Domain.");
    if (mxGetVectorLength(W_b_in) != numMaterials)
        mexErrMsgTxt("W_b must be of size equal to # of materials in Domain.");
    if (mxGetVectorLength(T_b_in) != numMaterials)
        mexErrMsgTxt("T_b must be of size equal to # of materials in Domain.");
    if (!mxAreSameDimensions(Domain_in, T0_in))
        mexErrMsgTxt("T0 must be of the same size as Domain.");

    /* Get problem size by checking Domain's dimensions */
    int Nx = mxGetDimensions(Domain_in)[1];
    int Ny = mxGetDimensions(Domain_in)[0];
    int Nz = mxGetDimensions(Domain_in)[2];

    /* Call the main bioheat solver function (which will save the results to T1) and output result*/
    plhs[0] = mxDuplicateArray(T0_in); //output T1 array (copy of T0)
    double *T1 = (double*)mxGetData(plhs[0]); //T1 is a copy of T0 used for calculations in solveBioheat and to store result
    solveBioheat(Domain, Qr, Qt, dh, dt, Nt, kappa, Ct, rho, Ct_b, rho_b, W_b, T_b, T0, T1, Nx, Ny, Nz, numMaterials);
}