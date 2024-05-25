#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

/*
并行思路：将图像分块,每个线程块处理一个块，每个线程处理一个像素点，每个线程块的像素点之间进行并行处理，最后处理边界
1. 读取图像数据，检查图像的尺寸，根据图像的尺寸计算出线程块的数量和每个线程块的大小
2. 在GPU上分配内存，将图像数据拷贝到GPU上
3. 在GPU上初始化并查集结构
4. 在GPU上进行并查集的合并操作
5. 检查每个线程块的边界，将边界的像素点进行合并操作，并更新像素内部的像素点的值
6. 将结果拷贝回CPU，写入文件

*/

typedef struct {
    int* p;        // Each pixel's ancestor node
    int* image;    // Image data
    int rows;      // Number of rows in the image
    int cols;      // Number of columns in the image
} UnionFind;

// CUDA kernel for initializing the Union Find structure
__global__ void init_kernel(int* p, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * cols + x;
    if (x < cols && y < rows) {
        p[idx] = idx;  // Set each pixel to its own set
    }
}

// CUDA kernel for finding the ancestor of x with path compression
__device__ int find(int* p, int x) {
    if (p[x] != x) {
        p[x] = find(p, p[x]);
    }
    return p[x];
}

// Wrapper function for finding the ancestor of x with path compression
__device__ int find_compressed(int* p, int x) {
    int root = x;
    while (p[root] != root) {
        root = p[root];
    }
    while (x != root) {
        int newp = p[x];
        p[x] = root;
        x = newp;
    }
    return root;
}

// CUDA kernel for merging the sets of two pixels
__global__ void union_kernel(int* p, int* image, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * cols + x;

    if (x < cols && y < rows) {
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy < 1; dy++) {
                if (dx >= 0 && dy == 0) continue;
                int ni = y + dy, nj = x + dx;
                if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && image[ni * cols + nj] == image[idx]) {
                    int rootX = find_compressed(p, idx);
                    int rootY = find_compressed(p, ni * cols + nj);
                    if (rootX != rootY) {
                        p[rootY] = rootX;
                    }
                }
            }
        }
    }
}

// CUDA kernel for path compression after union operations
__global__ void compress_paths_kernel(int* p, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * cols + x;

    if (x < cols && y < rows) {
        find_compressed(p, idx);
    }
}

// Function to write the result of the Union Find to a file
void writeResultToFile(UnionFind* uf, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "%d %d\n", uf->rows, uf->cols);

    for (int i = 0; i < uf->rows; i++) {
        for (int j = 0; j < uf->cols; j++) {
            fprintf(fp, "%d ", uf->p[i * uf->cols + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    UnionFind uf;
    int rows, cols;
    FILE* input_file = fopen(input_filename, "r");
    if (input_file == NULL) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }

    fscanf(input_file, "%d %d", &rows, &cols);

    uf.rows = rows;
    uf.cols = cols;

    int* h_image = (int*)malloc(rows * cols * sizeof(int));
    int* h_p = (int*)malloc(rows * cols * sizeof(int));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(input_file, "%d", &h_image[i * cols + j]);
        }
    }

    fclose(input_file);

    int* d_image;
    int* d_p;

    cudaMalloc(&d_image, rows * cols * sizeof(int));
    cudaMalloc(&d_p, rows * cols * sizeof(int));

    cudaMemcpy(d_image, h_image, rows * cols * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    init_kernel<<<gridSize, blockSize>>>(d_p, rows, cols);
    cudaDeviceSynchronize();

    union_kernel<<<gridSize, blockSize>>>(d_p, d_image, rows, cols);
    cudaDeviceSynchronize();

    compress_paths_kernel<<<gridSize, blockSize>>>(d_p, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_p, d_p, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    uf.p = h_p;
    uf.image = h_image;

    writeResultToFile(&uf, output_filename);

    free(h_image);
    free(h_p);
    cudaFree(d_image);
    cudaFree(d_p);

    return 0;
}
