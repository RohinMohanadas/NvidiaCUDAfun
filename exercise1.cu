#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using  namespace std;
__global__
void printHello(){
    printf("Hello World! My threadId is %d\n",threadIdx.x);
    //printf("I am a part of block : %d\n",blockIdx.x);
}

int main(){
    //int n = 256;
    printHello<<<1,256>>>();
    //cudaDeviceSynchronize();
    exit(0);
}