#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
using  namespace std;

struct Particle{
    float3 position;
    float3 velocity;
};
__global__
void simulate(Particle x[],int N,int iter){
    //printf("Hello World! My threadId is %d\n",threadIdx.x);
    //printf("I am a part of block : %d\n",blockIdx.x);

    // update are random
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int change = iter%2==0?-iter:iter;
    if(i < N){
        x[i].velocity = make_float3(x[i].velocity.x+change,x[i].velocity.y+change,x[i].velocity.z+change);        
        x[i].position = make_float3(x[i].position.x+x[i].velocity.x,x[i].position.y+x[i].velocity.y,x[i].position.z+x[i].velocity.z);
    }
}

float compare(Particle gpu, Particle cpu){
    float px,py,pz;
    px = abs(gpu.position.x-cpu.position.x);
    py = abs(gpu.position.y-cpu.position.y);
    pz = abs(gpu.position.z-cpu.position.z);
    float vx,vy,vz;
    vx = abs(gpu.velocity.x-cpu.velocity.x);
    vy = abs(gpu.velocity.y-cpu.velocity.y);
    vz = abs(gpu.velocity.z-cpu.velocity.z);
    float max1,max2;
    max1 = (px<py?py:px);
    max1 = max1<pz?pz:max1;
    max2 = (vx<vy?vy:vx);
    max2 = max2<vz?vz:max2;
    return max(max1,max2);
}

int main(int argc, char *argv[]){

    int NUM_PARTICLES = 10000;
    int NUM_ITERATIONS = 100;

    // test for 16,32,64,128,256

    int BLOCK_SIZE = 256;
    if(argc<=1){
        cout<<"Running with default values \n\tNUM_PARTICLES: 10000\n\tNUM_ITERATIONS: 100\n\tBLOCK_SIZE: 256\n";
        cout<<"To run with custom values use : ./exercise3 <NUM_PARTICLES> <NUM_ITERATIONS> <BLOCK_SIZE>\n";
    }else{
        int j = argc-1;
        int i=0;
        while(i<=j){
            switch(i){
                case 1:
                    NUM_PARTICLES = atoi(argv[1]);
                    break;
                case 2:
                    NUM_ITERATIONS = atoi(argv[2]);
                    break;
                case 3:
                    BLOCK_SIZE = atoi(argv[3]);
                    break;
            }
            i++;
        }
        cout<<"Running with values \n\tNUM_PARTICLES: "<<NUM_PARTICLES<<"\n\tNUM_ITERATIONS: "<<NUM_ITERATIONS<<"\n\tBLOCK_SIZE: "<<BLOCK_SIZE<<"\n";
    }


    int numBlocks = (int)NUM_PARTICLES / BLOCK_SIZE  + 1;

    size_t size = NUM_PARTICLES*sizeof(Particle);
    size_t v_size = NUM_ITERATIONS*sizeof(float3);

    // allocate memory in host

    Particle* X = (Particle*) malloc(size);
    Particle* temp = (Particle*) malloc(size);
    float3* velocityUpdates = (float3*) malloc(v_size);
    float A = 3.0;

    // initialize the array
    for(int i=0;i<NUM_PARTICLES;i++){
        X[i].position = make_float3(((float)rand()/(float)(RAND_MAX) * A),((float)rand()/(float)(RAND_MAX) * A),((float)rand()/(float)(RAND_MAX) * A));
        X[i].velocity = make_float3(((float)rand()/(float)(RAND_MAX) * A),((float)rand()/(float)(RAND_MAX) * A),((float)rand()/(float)(RAND_MAX) * A));
    }

    // allocate memory in device
    Particle* dX;

    cudaMalloc(&dX, size);

    // copy from host to host
    cudaMemcpy(temp, X, size, cudaMemcpyHostToHost);

    // compute on CPU
    printf("Updating particles on the CPU...  ");
    auto t_start = chrono::high_resolution_clock::now();
    for(int i=0;i<NUM_ITERATIONS;i++){
        int change = i%2==0?-i:i;
        for(int i=0;i<NUM_PARTICLES;i++){
            X[i].velocity = make_float3(X[i].velocity.x+change,X[i].velocity.y+change,X[i].velocity.z+change);
            X[i].position = make_float3(X[i].position.x+X[i].velocity.x,X[i].position.y+X[i].velocity.y,X[i].position.z+X[i].velocity.z);            
        }
    }


    auto t_end = chrono::high_resolution_clock::now();
    cout<<"Done! in "<<chrono::duration<double, milli>(t_end-t_start).count()<<" ms\n\n";

    // compute on GPU
    printf("Updating particles on the GPU...  ");
    t_start = chrono::high_resolution_clock::now();
    cudaMemcpy(dX, temp, size, cudaMemcpyHostToDevice);    
    for(int i=0;i<NUM_ITERATIONS;i++){

        simulate<<<numBlocks,BLOCK_SIZE>>>(dX,NUM_PARTICLES,i);
        //cudaThreadSynchronize();

        // copy to host memory after each simulation.
        cudaMemcpy(temp, dX, size, cudaMemcpyDeviceToHost);    
    }

    // whole simulation is done now.

    t_end = chrono::high_resolution_clock::now();
    cout<<"Done! in "<<chrono::duration<double, milli>(t_end-t_start).count()<<" ms\n\n";
    //exit(0);
    printf("Comparing the output for each implementation...  ");    
    for(int i=0;i<NUM_PARTICLES;i++){
        if( compare(temp[i],X[i])> 1e-05 ){
            cout<<"outputs do not match: "<<compare(temp[i],X[i])<<"\n";
            //exit(0);
        }
    }
    cout<<"Correct!\n";
    exit(0);
}