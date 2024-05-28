#include "cclcuda.cuh"
#include <cmath>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>

const int BLOCK = 8;

__device__ int IMin(int a, int b)
{
    return a < b ? a : b;
}

__global__ void InitCCL(int labelList[], int reference[], int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int id = x + y * width;

    labelList[id] = reference[id] = id;
}

__global__ void scanning8(unsigned char frame[], int labelList[], int reference[], bool* markFlag, int N, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int id = x + y * width;

    if (id >= N)
        return;

    unsigned char value = frame[id];
    int label = N;

    if (id - width >= 0 && value == frame[id - width])
        label = IMin(label, labelList[id - width]);
    if (id + width < N  && value == frame[id + width])
        label = IMin(label, labelList[id + width]);

    int col = id % width;
    if (col > 0)
    {
        if (value == frame[id - 1])
            label = IMin(label, labelList[id - 1]);
        if (id - width - 1 >= 0 && value == frame[id - width - 1])
            label = IMin(label, labelList[id - width - 1]);
        if (id + width - 1 < N  && value == frame[id + width - 1])
            label = IMin(label, labelList[id + width - 1]);
    }
    if (col + 1 < width)
    {
        if (value == frame[id + 1])
            label = IMin(label, labelList[id + 1]);
        if (id - width + 1 >= 0 && value == frame[id - width + 1])
            label = IMin(label, labelList[id - width + 1]);
        if (id + width + 1 < N  && value == frame[id + width + 1])
            label = IMin(label, labelList[id + width + 1]);
    }

    if (label < labelList[id])
    {
        reference[labelList[id]] = label;
        *markFlag = true;
    }
}

__global__ void analysis(int labelList[], int reference[], int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int id = x + y * width;

    int label = labelList[id];
    int ref;
    if (label == id)
    {
        do
        {
            ref = label;
            label = reference[ref];
        } while (ref ^ label);
        reference[id] = label;
    }
}

__global__ void labeling(int labelList[], int reference[], int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int id = x + y * width;

    labelList[id] = reference[reference[labelList[id]]];
}

void CCLLEGPU::CudaCCL(unsigned char* frame, int* labels, int width, int height, int degreeOfConnectivity, unsigned char threshold)
{
    auto N = width * height;

    cudaMalloc(reinterpret_cast<void**>(&LabelListOnDevice), sizeof(int) * N);
    cudaMalloc(reinterpret_cast<void**>(&ReferenceOnDevice), sizeof(int) * N);
    cudaMalloc(reinterpret_cast<void**>(&FrameDataOnDevice), sizeof(unsigned char) * N);

    cudaMemcpy(FrameDataOnDevice, frame, sizeof(unsigned char) * N, cudaMemcpyHostToDevice);

    bool* markFlagOnDevice;
    cudaMalloc(reinterpret_cast<void**>(&markFlagOnDevice), sizeof(bool));

    dim3 grid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
    dim3 threads(BLOCK, BLOCK);

    InitCCL <<<grid, threads >>>(LabelListOnDevice, ReferenceOnDevice, width, height);

    while (true)
    {
        auto markFalgOnHost = false;
        cudaMemcpy(markFlagOnDevice, &markFalgOnHost, sizeof(bool), cudaMemcpyHostToDevice);

        scanning8 <<< grid, threads >>>(FrameDataOnDevice, LabelListOnDevice, ReferenceOnDevice, markFlagOnDevice, N, width, height);

        cudaDeviceSynchronize();
        cudaMemcpy(&markFalgOnHost, markFlagOnDevice, sizeof(bool), cudaMemcpyDeviceToHost);

        if (markFalgOnHost)
        {
            analysis <<< grid, threads >>>(LabelListOnDevice, ReferenceOnDevice, width, height);
            cudaDeviceSynchronize();
            labeling <<< grid, threads >>>(LabelListOnDevice, ReferenceOnDevice, width, height);
        }
        else
        {
            break;
        }
    }

    cudaMemcpy(labels, LabelListOnDevice, sizeof(int) * N, cudaMemcpyDeviceToHost);

    cudaFree(FrameDataOnDevice);
    cudaFree(LabelListOnDevice);
    cudaFree(ReferenceOnDevice);
}
