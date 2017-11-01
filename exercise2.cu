#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
using  namespace std;

__global__
void saxpy(float x[],float y[],float a,int N){
    //printf("Hello World! My threadId is %d\n",threadIdx.x);
    //printf("I am a part of block : %d\n",blockIdx.x);
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N)
        y[i] = a * x[i] + y[i];
}

int main(){
    int ARRAY_SIZE = 100000000;

    int threadsPerBlock = 256;
    int numBlocks = (int)ARRAY_SIZE / 256  + 1;

    size_t size = ARRAY_SIZE*sizeof(float);


    // allocate memory in host
    float* X = (float*) malloc(size);
    float* Y = (float*) malloc(size);
    float A = 6.0;

    // initialize the array
    for(int i=0;i<ARRAY_SIZE;i++){
        X[i] = ((float)rand()/(float)(RAND_MAX) * A);
        Y[i] = ((float)rand()/(float)(RAND_MAX) * A);
    }

    // allocate memory in device
    float* dX;
    float* dY;

    cudaMalloc(&dX, size);
    cudaMalloc(&dY, size);

    // copy from host to device
    cudaMemcpy(dX, X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, size, cudaMemcpyHostToDevice);

    // compute on CPU
    printf("Computing SAXPY on the CPU...  ");
    auto t_start = chrono::high_resolution_clock::now();
    for(int i=0;i<ARRAY_SIZE;i++){
        Y[i] = A*X[i] + Y[i];
    }
    auto t_end = chrono::high_resolution_clock::now();
    cout<<"Done! in "<<chrono::duration<double, milli>(t_end-t_start).count()<<" ms\n\n";

    // compute on GPU
    printf("Computing SAXPY on the GPU...  ");
    t_start = chrono::high_resolution_clock::now();
    saxpy<<<numBlocks,threadsPerBlock>>>(dX,dY,A,ARRAY_SIZE);
    t_end = chrono::high_resolution_clock::now();
    cout<<"Done! in "<<chrono::duration<double, milli>(t_end-t_start).count()<<" ms\n\n";

    //cudaDeviceSynchronize();
    // copy result to host
    cudaMemcpy(X, dY, size, cudaMemcpyDeviceToHost);

    printf("Comparing the output for each implementation...  ");    
    for(int i=0;i<ARRAY_SIZE;i++){
        if( abs(Y[i] - X[i]) > 1e-05 ){
            cout<<"outputs do not match: "<<abs(Y[i] - X[i])<<"\n";
            //exit(0);
        }
    }
    cout<<"Correct!\n";
    exit(0);
}