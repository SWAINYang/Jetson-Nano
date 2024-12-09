#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define PI 3.1415926 // Define the value of pi
#define FILENAME "im.pgm" // Define the name of the PGM image file

// Define a structure to store information about the PGM image
typedef struct {
    int width; // Width of the image
    int height; // Height of the image
    int maxval; // Maximum grayscale value of the image
    unsigned char* data; // Pixel data of the image stored in a one-dimensional array
} PGMImage;

// Function to read a PGM image file and store its information in a PGMImage structure
void readPGM(PGMImage* image, const char* filename) {
    FILE* fp = fopen(filename, "r"); // Open the file in text mode
    if (fp == NULL) { // If the file opening fails, print an error message and exit the program
        perror("Cannot open file to read");
        exit(EXIT_FAILURE);
    }

    char ch; // Variable to store characters from the file
    int i; // Loop counter

    // Read the first line of the file and check if it is the P2 identifier
    if (fscanf(fp, "%c%c", &ch, &ch) != 2) {
        fprintf(stderr, "Error reading file header\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    if (ch != '2') { // If it is not the P2 identifier, print an error message and exit the program
        fprintf(stderr, "Not a valid P2 PGM file\n");
        exit(EXIT_FAILURE);
    }

    // Skip the newline character after the first line of the file
    fgetc(fp);

    // Skip comment lines in the file, if any
    while ((ch = fgetc(fp)) == '#') {
        while (fgetc(fp) != '\n');
    }

    // Put the last read character back into the file stream for later use with fscanf
    ungetc(ch, fp);

    // Read the width, height, and maximum grayscale value of the image from the file
    if (fscanf(fp, "%d%d%d", &image->width, &image->height, &image->maxval) != 3) {
        fprintf(stderr, "Error reading image size and maxval\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Dynamically allocate memory for storing the pixel data based on the width and height of the image
    image->data = (unsigned char*)malloc(image->width * image->height * sizeof(unsigned char));
    if (image->data == NULL) {
        fprintf(stderr, "Failed to allocate memory for pixel data\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Read the pixel data of the image from the file and store it in a one-dimensional array
    for (i = 0; i < image->width * image->height; i++) {
        if (fscanf(fp, "%hhu", &image->data[i]) != 1) {
            fprintf(stderr, "Error reading pixel data\n");
            fclose(fp);
            free(image->data);
            exit(EXIT_FAILURE);
        }
    }

    // Close the file
    fclose(fp);
}


// Function to write a PGM image file, writing the information from a PGMImage structure to the file
void writePGM(PGMImage* image, const char* filename) {
    FILE* fp = fopen(filename, "w"); // Open the file in text mode
    if (fp == NULL) { // If the file opening fails, print an error message and exit the program
        perror("Cannot open file to write");
        exit(EXIT_FAILURE);
    }

    int i; // Loop counter

    // Write the P2 identifier to the first line of the file
    fprintf(fp, "P2\n");

    // Write the width and height of the image to the second line of the file
    fprintf(fp, "%d %d\n", image->width, image->height);

    // Write the maximum grayscale value of the image to the third line of the file
    fprintf(fp, "%d\n", image->maxval);

    // Write the pixel data of the image to the file, separating each value with a space
    for (i = 0; i < image->width * image->height; i++) {
        fprintf(fp, "%hhu ", image->data[i]);
    }

    // Close the file
    fclose(fp);
}

__global__ void rotatePGMKernel(unsigned char* src, unsigned char* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight, double cosine, double sine, double ori_centre_x, double ori_centre_y, double new_centre_x, double new_centre_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dstWidth && y < dstHeight) {
        double sx = (y - new_centre_y) * sine + (x - new_centre_x) * cosine + ori_centre_x;
        double sy = (y - new_centre_y) * cosine - (x - new_centre_x) * sine + ori_centre_y;

        if (sx >= 0 && sx < srcWidth && sy >= 0 && sy < srcHeight) {
            int x1 = (int)(sx + 0.5);
            int y1 = (int)(sy + 0.5);
            dst[y * dstWidth + x] = src[y1 * srcWidth + x1];
        } else {
            dst[y * dstWidth + x] = 0;
        }
    }
}

void rotatePGM(PGMImage* src, PGMImage* dst, double angle) {
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing the rotatePGM function
    cudaEventRecord(start);

    double radian = angle * PI / 180;
    double cosine = cos(radian);
    double sine = sin(radian);

    double ori_centre_x = (src->width - 1) / 2.0;
    double ori_centre_y = (src->height - 1) / 2.0;

    dst->height = (int)ceil(fabs(src->height * cosine) + fabs(src->width * sine));
    dst->width = (int)ceil(fabs(src->width * cosine) + fabs(src->height * sine));

    double new_centre_x = (dst->width - 1) / 2.0;
    double new_centre_y = (dst->height - 1) / 2.0;

    dst->maxval = src->maxval;
    dst->data = (unsigned char*)malloc(dst->width * dst->height * sizeof(unsigned char));

    unsigned char *d_src, *d_dst;
    size_t srcSize = src->width * src->height * sizeof(unsigned char);
    size_t dstSize = dst->width * dst->height * sizeof(unsigned char);

    cudaMalloc(&d_src, srcSize);
    cudaMalloc(&d_dst, dstSize);

    cudaMemcpy(d_src, src->data, srcSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((dst->width + 31) / 32, (dst->height + 31) / 32);
    rotatePGMKernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, src->width, src->height, dst->width, dst->height, cosine, sine, ori_centre_x, ori_centre_y, new_centre_x, new_centre_y);

    cudaMemcpy(dst->data, d_dst, dstSize, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);

    // Stop timing the rotatePGM function
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time for rotatePGM function
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("rotatePGM execution time: %f ms\n", milliseconds);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main() {
    // Create CUDA events for timing the entire program
    cudaEvent_t startProgram, stopProgram;
    cudaEventCreate(&startProgram);
    cudaEventCreate(&stopProgram);

    // Start timing the entire program
    cudaEventRecord(startProgram);

    PGMImage src, dst;
    double angle = 45; // Example rotation angle, replace with desired value or argument

    // Read the source image
    readPGM(&src, FILENAME);

    // Perform the rotation
    rotatePGM(&src, &dst, angle);

    // Write the rotated image to a file
    writePGM(&dst, "rotated_im.pgm");

    // Free allocated memory
    free(src.data);
    free(dst.data);

    // Stop timing the entire program
    cudaEventRecord(stopProgram);
    cudaEventSynchronize(stopProgram);

    // Calculate and print the elapsed time for the entire program
    float totalMilliseconds = 0;
    cudaEventElapsedTime(&totalMilliseconds, startProgram, stopProgram);
    printf("Total program execution time: %f ms\n", totalMilliseconds);

    // Clean up CUDA events
    cudaEventDestroy(startProgram);
    cudaEventDestroy(stopProgram);

    return 0;
}





