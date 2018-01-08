/*
This program calculates the pressure in a volume caused by a vibrating boundary, by Thomas Watson 2017

Modified from code by Sebastian Maurino, 2016
Based on Nicholas Ellens' 2011 ForwardTransmission solver

[pressure] = pressureinvolume(k, c, rho, Tx_center, Tx_ds, Tx_norm, Tx_u, x, y, z, mask)

k:          Complex wavenumber of waves in medium                   complex double scalar
c:          Speed of waves (m/s) in medium                          double scalar
rho:        Density (kg/m^3) of medium                              double scalar
Tx_center:  Center coordinates of transducer subelements            double matrix of size [NTx, 3]
Tx_ds:      Area of each transducer subelement                      double column vector of size [NTx]
Tx_norm:    Norm coordinates of transducer subelements              double matrix of size [NTx, 3]
Tx_u:       Complex velocity of each transducer element             complex double column vector of size [NTx]
x:          Values of x-dimension of each voxel in cubic volume     double column vector od size [Nx]
y:          Values of y-dimension of each voxel in cubic volume     double column vector od size [Ny]
z:          Values of z-dimension of each voxel in cubic volume     double column vector od size [Nz]
mask:       Subset of voxels in volume correspoding to medium       logical matrix of size [Ny, Nx, Nz]


*/

#include <algorithm>
#include <thrust/complex.h>
#include "cudaatomicaddcomplex.hpp" // Needed to write to same address from multiple 
#include <glm/vec3.hpp> // 3d vectors (glm::tvec3<T>)
#include <glm/geometric.hpp> 
#include "mex.h"
#include "matrix.h"
#include "isMaskvisible.hpp"

using namespace std;

// Constant for threads per block to be used by CUDA. 256 is the standard
const int THREADS_PER_BLOCK = 256;


// Cuda kernel function - This function will be executed many times in parallel in the GPU. It uses the value of the thread ID to know which voxel to evaluate ----------------------------------------

__global__ void pressure_voxel_Tx(thrust::complex<double> k, double c, double rho, double *Tx_center, double *Tx_ds, thrust::complex<double> *Tx_u, double *x, double *y, double *z, bool *mask,
                                  thrust::complex<double> *pressure, int ocl_mode, int NTx, int Nx, int Ny, int Nz, dim3 gridIdx)
{
    // Define the complex constant
    const thrust::complex<double> I = {0, 1} ;

    // Get voxel, transducer, and dimensional indices
    int ind = threadIdx.x + (blockIdx.x*blockDim.x) + (gridIdx.x*gridDim.x*blockDim.x) ;
    int iTx = threadIdx.y + (blockIdx.y*blockDim.y) + (gridIdx.y*gridDim.y*blockDim.y) ;
    int iz = ind/(Nx*Ny) ;
    int ix = (ind%(Nx*Ny))/Ny ;
    int iy = (ind%(Nx*Ny))%Ny ;

    // Check if voxel and transducer indices are inside problem space
    if (iz >= Nz || iTx >= NTx)
        return ;

    // Check if input velocity is zero and return if true
    if (Tx_u[iTx] == 0.0)
        return ;

    // Check if voxel is in mask and return if false
    if (!mask[ind])
        return ;

    // Define target voxel at which pressure is presently being evaluated
    bool vis = true ;

    // Check if voxel is visible from mask if occlusion mode is on using mask based occlusion
    switch(ocl_mode)
    {
        case 2:
        {
            glm::tvec3<int> finish_pos_Vox ;
            finish_pos_Vox.x = ix ;
            finish_pos_Vox.y = iy ;
            finish_pos_Vox.z = iz ;
            glm::tvec3<double> Tx_pos ;
            Tx_pos.x = Tx_center[iTx] ;
            Tx_pos.y = Tx_center[iTx + NTx] ;
            Tx_pos.z = Tx_center[iTx + 2*NTx] ;
            vis = isVisiblePresMask(Tx_pos, finish_pos_Vox, mask, Nx, Ny, Nz, x, y, z) ;
            break ;
        }
    }
    if (!vis)
        return ; // Face is occluded


    // Calculate the distance, R, between Tx_pos and finish_pos_Vox
    double Delta_x = x[ix] - Tx_center[iTx] ;
    double Delta_y = y[iy] - Tx_center[iTx + NTx] ;
    double Delta_z = z[iz] - Tx_center[iTx + 2*NTx] ;
    double R = sqrt(pow(Delta_x, 2) + pow(Delta_y, 2) + pow(Delta_z, 2)) ;

    // Use epsilon to check if voxel is near boundary
    double epsilon = sqrt(Tx_ds[iTx]/6.283185308);//cbrt(3*Tx_ds[iTx] / (2*3.141592654*k.real())) ;

    // Calculate the addition or subtraction to voxel pressure according to epsilon (O'Neil 49)
    thrust::complex<double> dp, dpx, dpy, dpz ;
    if (R > epsilon)
        dp = (rho*c/6.283185308) * I * k.real() * (thrust::exp( - I * k * R ) / R) * Tx_u[iTx] * Tx_ds[iTx] ;
    else
        dp = 3*rho*c*I*Tx_u[iTx]*k.real()*sqrt(Tx_ds[iTx])*thrust::exp(-I*k*R)/(4*sqrt(6.283185308)) ; //(thrust::exp( - I * k * epsilon )*(1.0 + I*k.real()*epsilon) - 1.0 ) ;
    
    // Update pressure
    atomicAdd(&(pressure[ind]), dp) ;
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








// Main Solver ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void solvePressure(thrust::complex<double> k, double c, double rho, double *Tx_center, double *Tx_ds, thrust::complex<double> *Tx_u, double *x, double *y, double *z, bool *mask,
                   double *pressure_real, double *pressure_imag, int ocl_mode, int NTx, int Nx, int Ny, int Nz)
{
    // Output GPU details to user
    int device ;
    cudaGetDevice(&device) ;
    cudaDeviceProp properties ;
    cudaGetDeviceProperties(&properties, device) ;
    mexPrintf("Running Pressure in Volume solver in GPU device %s\n", properties.name) ;
    mexEvalString("drawnow") ; // Forces MATLAB to output
    cudaErrorCheck("initializing device") ;

    // Calculate maximum index of voxel
    int voxels = Nx * Ny * Nz ;

    // Allocate GPU arrays
    double *d_Tx_center ;
    size_t sz_3NTx_double = 3*NTx * sizeof(double) ;
    cudaMalloc(&d_Tx_center, sz_3NTx_double) ;

    double *d_Tx_ds ;
    size_t sz_NTx_double = NTx * sizeof(double) ;
    cudaMalloc(&d_Tx_ds, sz_NTx_double) ;

    thrust::complex<double> *d_Tx_u ;
    size_t sz_NTx_complex = NTx * sizeof(thrust::complex<double>) ;
    cudaMalloc(&d_Tx_u, sz_NTx_complex) ;

    double *d_x ;
    size_t sz_Nx_double = Nx * sizeof(double) ;
    cudaMalloc(&d_x, sz_Nx_double) ;

    double *d_y ;
    size_t sz_Ny_double = Ny * sizeof(double) ;
    cudaMalloc(&d_y, sz_Ny_double) ;

    double *d_z ;
    size_t sz_Nz_double = Nz * sizeof(double) ;
    cudaMalloc(&d_z, sz_Nz_double) ;

    bool *d_mask ;
    size_t sz_voxels_bool = voxels * sizeof(bool) ;
    cudaMalloc(&d_mask, sz_voxels_bool) ;

    thrust::complex<double> *d_pressure ;
    size_t sz_voxels_complex = voxels * sizeof(thrust::complex<double>) ;
    cudaMalloc(&d_pressure, sz_voxels_complex) ;

    cudaErrorCheck("allocating memory") ;

    // Copy arrays to Device
    cudaMemcpy(d_Tx_center, Tx_center, sz_3NTx_double, cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_Tx_ds, Tx_ds, sz_NTx_double, cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_Tx_u, Tx_u, sz_NTx_complex, cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_x, x, sz_Nx_double, cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_y, y, sz_Ny_double, cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_z, z, sz_Nz_double, cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_mask, mask, sz_voxels_bool, cudaMemcpyHostToDevice) ;
    cudaMemset(d_pressure, 0, sz_voxels_complex) ;

    cudaErrorCheck("copying memory to device") ;

    // Calculate size of GPU grid, blocks, and threads
    // Split into separate chunks if grid size is larger than maximum
    // x: Voxel
    // y: Transducer subelement
    int THREADS_PER_BLOCK_XY = sqrt(THREADS_PER_BLOCK) ;
    dim3 numOfGrids(((voxels/THREADS_PER_BLOCK_XY )/properties.maxGridSize[0] ) + 1, ((NTx/THREADS_PER_BLOCK_XY)/properties.maxGridSize[1]) + 1, 1) ;
    dim3 blocksPerGrid((std::min(voxels/THREADS_PER_BLOCK_XY + 1, properties.maxGridSize[0])), (std::min(NTx/THREADS_PER_BLOCK_XY + 1, properties.maxGridSize[1])), 1) ;
    dim3 threadsPerBlock(THREADS_PER_BLOCK_XY, THREADS_PER_BLOCK_XY, 1) ;

    // Run the Kernels in the GPU
    // Each grid chunk is run in serial to avoid surpassing the maximum grid size
    for (int gx = 0; gx < numOfGrids.x; gx++)
    {
        for (int gy = 0; gy < numOfGrids.y; gy++)
        {
            // Evaluate the contributions of each transducer at each voxel
            dim3 gridIdx(gx, gy, 1) ;
            pressure_voxel_Tx <<< blocksPerGrid, threadsPerBlock >>>
            (   k, c, rho, d_Tx_center, d_Tx_ds, d_Tx_u, d_x, d_y, d_z, d_mask, d_pressure, ocl_mode, NTx, Nx, Ny, Nz, gridIdx
            ) ;

            cudaDeviceSynchronize() ; // Wait for all device calls to complete
        }
    }

    cudaErrorCheck("calling kernels") ; // Check for errors in kernel calls


    // Copy results back to Host
    thrust::complex<double> *temp_result = new thrust::complex<double>[voxels] ;
    cudaMemcpy(temp_result, d_pressure, sz_voxels_complex, cudaMemcpyDeviceToHost) ;
    for (int i = 0; i < voxels; i++)
    {
        pressure_real[i] = temp_result[i].real() ;
        pressure_imag[i] = temp_result[i].imag() ;
    }
    cudaErrorCheck("copying memory to host") ;



    // Free GPU buffers
    cudaFree(d_Tx_center) ;
    cudaFree(d_Tx_ds) ;
    cudaFree(d_Tx_u) ;
    cudaFree(d_x) ;
    cudaFree(d_y) ;
    cudaFree(d_z) ;
    cudaFree(d_mask) ;
    cudaFree(d_pressure) ;

    cudaErrorCheck("freeing memory") ;

    cudaDeviceReset() ;
}


// MATLAB Interfacing and Data Validating Section -----------------------------------------------------------------------------------------------------------------------------------------------------

bool mxAreSameDimensions(const mxArray *arrayPtr1, const mxArray *arrayPtr2)
{
    // Check that they have the same number of dimensions
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
{ //this function checks for correct inputs, casts them to the correct data types and calls the main solver function

    /* Check for the correct number of input and output arguments*/
    if (nrhs != 11)
        mexErrMsgTxt("Exactly 11 input arguments are required.\nThe correct syntax is [pressure] = pressureinvolume(k, c, rho, Tx_center, Tx_ds, Tx_u, x, y, z, mask, ocl_mode)");
    if (nlhs != 1)
        mexErrMsgTxt("Exactly 1 output argument is required.\nThe correct syntax is [pressure] = pressureinvolume(k, c, rho, Tx_center, Tx_ds, Tx_u, x, y, z, mask, ocl_mode)");


    /* Set right hand side parameter pointers to more readable names */
    const mxArray *k_in = prhs[0], *c_in = prhs[1], *rho_in = prhs[2], *Tx_center_in = prhs[3], *Tx_ds_in = prhs[4], *Tx_u_in = prhs[5], *x_in = prhs[6], *y_in = prhs[7], *z_in = prhs[8], *mask_in = prhs[9], *ocl_mode_in = prhs[10];


    /* Check for the correct data types and sizes */
    if (!mxIsDouble(k_in) || !mxIsComplex(k_in) || !mxIsScalar(k_in))
        mexErrMsgTxt("k must be a complex double scalar");
    if (!mxIsDouble(c_in) || !mxIsScalar(c_in))
        mexErrMsgTxt("c must be a double scalar");
    if (!mxIsDouble(rho_in) || !mxIsScalar(rho_in))
        mexErrMsgTxt("rho must be a double scalar");
    if (!mxIsDouble(Tx_center_in) || mxIsScalar(Tx_center_in) || mxGetN(Tx_center_in) != 3)
        mexErrMsgTxt("Tx_center must be a double matrix of size [NTx, 3]");
    if (!mxIsDouble(Tx_ds_in) || mxIsScalar(Tx_ds_in) || !mxIsColumnVector(Tx_ds_in) || mxGetVectorLength(Tx_center_in) != mxGetVectorLength(Tx_ds_in))
        mexErrMsgTxt("Tx_ds must be a double column vector of size [NTx]");
    if (!mxIsDouble(Tx_u_in) || !mxIsComplex(k_in) || mxIsScalar(Tx_u_in) || !mxIsColumnVector(Tx_u_in) || mxGetVectorLength(Tx_center_in) != mxGetVectorLength(Tx_u_in))
        mexErrMsgTxt("Tx_u must be a complex double column vector of size [NTx]");
    if (!mxIsDouble(x_in) || mxIsScalar(x_in) || !mxIsColumnVector(x_in))
        mexErrMsgTxt("x must be a double column vector");
    if (!mxIsDouble(y_in) || mxIsScalar(y_in) || !mxIsColumnVector(y_in))
        mexErrMsgTxt("y must be a double column vector");
    if (!mxIsDouble(z_in) || mxIsScalar(z_in) || !mxIsColumnVector(z_in))
        mexErrMsgTxt("z must be a double column vector");
    if (!mxIsLogical(mask_in) || mxIsScalar(mask_in) || mxGetNumberOfDimensions(mask_in) != 3 || mxGetDimensions(mask_in)[0] != mxGetVectorLength(y_in) || mxGetDimensions(mask_in)[1] != mxGetVectorLength(x_in) || mxGetDimensions(mask_in)[2] != mxGetVectorLength(z_in))
        mexErrMsgTxt("mask must be a logical matrix of size [Ny, Nx, Nz]");
    if (!mxIsInt32(ocl_mode_in))
        mexErrMsgTxt("ocl_mode must be an integer");
    if (!mxIsScalar(ocl_mode_in))
        mexErrMsgTxt("ocl_mode must be a scalar");


    /* Get problem size */
    int Nx = mxGetVectorLength(x_in);
    int Ny = mxGetVectorLength(y_in);
    int Nz = mxGetVectorLength(z_in);
    int NTx = mxGetVectorLength(Tx_center_in);


    /* Load all data */
    thrust::complex<double> k, *Tx_u = new thrust::complex<double> [NTx];
    double c, rho, *Tx_center, *Tx_ds, *x, *y, *z;
    bool *mask;

    k = thrust::complex<double>( *mxGetPr(k_in), *mxGetPi(k_in) );
    c = mxGetScalar(c_in);
    rho = mxGetScalar(rho_in);
    Tx_center = (double*) mxGetData(Tx_center_in);
    Tx_ds = (double*) mxGetData(Tx_ds_in);
    for (int i = 0; i < NTx; i++)
        Tx_u[i] = thrust::complex<double> ( mxGetPr(Tx_u_in)[i], mxGetPi(Tx_u_in)[i] );
    x = (double*) mxGetData(x_in);
    y = (double*) mxGetData(y_in);
    z = (double*) mxGetData(z_in);
    mask = mxGetLogicals(mask_in);
    int ocl_mode = (int)mxGetScalar(ocl_mode_in);


    /* Create output array */
    plhs[0] = mxCreateNumericArray(3, mxGetDimensions(mask_in), mxDOUBLE_CLASS, mxCOMPLEX);
    double *pressure_real, *pressure_imag;
    pressure_real = (double*) mxGetPr(plhs[0]);
    pressure_imag = (double*) mxGetPi(plhs[0]);

    solvePressure(k, c, rho, Tx_center, Tx_ds, Tx_u, x, y, z, mask, pressure_real, pressure_imag, ocl_mode, NTx, Nx, Ny, Nz);

    delete[] Tx_u ;
}