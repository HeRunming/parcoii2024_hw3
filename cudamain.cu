#include <stdio.h>
#include <stdlib.h>
#include "cclcuda.cu"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iomanip>
#include <fstream>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    int rows, cols;
    FILE* input_file = fopen(input_filename, "r");
    if (input_file == NULL) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }

    fscanf(input_file, "%d %d", &rows, &cols);

    unsigned char data[rows * cols];

    for (int i = 0; i < rows * cols; i++) {
        fscanf(input_file, "%d", &data[i]);
    }

    fclose(input_file);

    int labels[rows * cols] = { 0 };

    auto degreeOfConnectivity = 4;
	unsigned char threshold = 0;

    CCLLEGPU ccl;

    ccl.CudaCCL(data, labels, cols, rows, degreeOfConnectivity, threshold);

    FILE* output_file = fopen(output_filename, "w");


    // write the output to the file
    fprintf(output_file, "%d %d\n", rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        fprintf(output_file, "%d ", labels[i]);
    }
}