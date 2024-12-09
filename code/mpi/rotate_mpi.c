#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#define PI 3.1415926
#define FILENAME "im.pgm"

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

void rotatePGM(PGMImage* src, PGMImage* dst, double angle, int rank, int size) {
    double radian = angle * PI / 180.0;
    double cosine = cos(radian);
    double sine = sin(radian);

    int ori_centre_x = src->width / 2;
    int ori_centre_y = src->height / 2;
    dst->width = (int)(fabs(src->width * cosine) + fabs(src->height * sine));
    dst->height = (int)(fabs(src->height * cosine) + fabs(src->width * sine));
    dst->maxval = src->maxval;

    if (rank == 0) {
        dst->data = (unsigned char*)malloc(dst->width * dst->height * sizeof(unsigned char));
        memset(dst->data, 0, dst->width * dst->height * sizeof(unsigned char));
    }

    int rows_per_process = (dst->height + size - 1) / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process;
    if (end_row > dst->height) end_row = dst->height;

    unsigned char* local_buffer = (unsigned char*)malloc(dst->width * rows_per_process * sizeof(unsigned char));
    memset(local_buffer, 0, dst->width * rows_per_process * sizeof(unsigned char));

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < dst->width; ++j) {
            int ori_x = (int)((i - dst->height / 2) * cosine - (j - dst->width / 2) * sine) + ori_centre_x;
            int ori_y = (int)((i - dst->height / 2) * sine + (j - dst->width / 2) * cosine) + ori_centre_y;

            if (ori_x >= 0 && ori_x < src->width && ori_y >= 0 && ori_y < src->height) {
                local_buffer[(i - start_row) * dst->width + j] = src->data[ori_y * src->width + ori_x];
            }
        }
    }

    MPI_Gather(local_buffer, dst->width * rows_per_process, MPI_UNSIGNED_CHAR,
               dst->data, dst->width * rows_per_process, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    free(local_buffer);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <rotation_angle>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    double angle = atof(argv[1]);
    PGMImage src, dst;

    if (rank == 0) {
        readPGM(&src, FILENAME);
    }

    MPI_Bcast(&src.width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&src.height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&src.maxval, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        src.data = (unsigned char*)malloc(src.width * src.height * sizeof(unsigned char));
    }

    MPI_Bcast(src.data, src.width * src.height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    double start_time, end_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    rotatePGM(&src, &dst, angle, rank, size);

    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Execution time: %f seconds\n", end_time - start_time);

        char new_filename[256];
        sprintf(new_filename, "rotated_%s", FILENAME);
        writePGM(&dst, new_filename);

        free(src.data);
        free(dst.data);
    }

    MPI_Finalize();
    return 0;
}











