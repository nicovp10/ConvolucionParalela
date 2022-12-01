#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define MASTER_RANK 0
#define K_ROWS 3
#define K_COLUMNS 3
#define MAX_COLUM_LENGTH 15000

#define ITER 1
#define C 8


int main(int argc, char *argv[]) {
    int myrank, mpi_size, a, i;
    struct timeval t1, t2;
    const char *f_in, *f_out;
    unsigned char *img_in, *img_out;
    int img_width, img_height;
    size_t img_size;
    MPI_Datatype col_type;
    float** kernel;
    unsigned char *data;

    
    // Initialization of MPI communication
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


    for (a = 0; a < ITER; a++) {
        // Allocate memory for the kernel
        kernel = malloc(K_ROWS * sizeof(float*));
        for (i = 0; i < K_ROWS; i++) {
            kernel[i] = malloc(K_COLUMNS * sizeof(float));
        }
        

        if (myrank == MASTER_RANK) {
            long total_blocks, send_blocks, num_cycles, rest;


            printf("Iteración: %d\n", a);


            // Set image filenames
            switch (argc) {
                case 2:
                    f_in = argv[1];
                    f_out = "output.jpg";
                    break;
                case 3:
                    f_in = argv[1];
                    f_out = argv[2];
                    break;
                default:
                    f_in = "input.jpg";
                    f_out = "output.jpg";
                    break;
            }

            // Load the input image
            img_in = stbi_load(f_in, &img_width, &img_height, NULL, 1);
            if(img_in == NULL) {
                printf("Error in loading the image\n");
                exit(EXIT_FAILURE);
            }
            img_size = img_width * img_height;

            // Allocate memory for the output image
            img_out = malloc(img_size);
            if(img_out == NULL) {
                printf("Unable to allocate memory for the convolved image.\n");
                exit(EXIT_FAILURE);
            }

            
            // Define the convolution kernel
            //      [0, -1, 0]
            //      [-1, 6, -1]
            //      [0, -1, 0]
            kernel[0][0] = 0;
            kernel[0][1] = -1;
            kernel[0][2] = 0;
            kernel[1][0] = -1;
            kernel[1][1] = 6;
            kernel[1][2] = -1;
            kernel[2][0] = 0;
            kernel[2][1] = -1;
            kernel[2][2] = 0;
        

            // Define the MPI Datatype column
            MPI_Type_vector(img_height, 1, img_width, MPI_UNSIGNED_CHAR, &col_type);
            MPI_Type_commit(&col_type);

            total_blocks = img_width / C;
            rest = total_blocks % mpi_size;
            num_cycles = total_blocks / mpi_size;
            if (num_cycles % mpi_size != 0) {
                num_cycles++;
            }

            // Send the corresponding colums to each process 
            send_blocks = 0;
            while (send_blocks != total_blocks) {
                //MPI_Send(img_in, 1, col_type, 1, 0, MPI_COMM_WORLD);
            }
        }


        // Receive the kernel from the master process
        MPI_Bcast(&(kernel[0][0]), K_ROWS*K_COLUMNS, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);


/*
        // First row
        for (unsigned char *p_in = img_in, *p_out = img_out;
                p_in != img_in + img_width;
                p_in += 1, p_out += 1) {
            *p_out = *p_in;
        }
        
        // First column
        for (unsigned char *p_in = img_in + img_width, *p_out = img_out + img_width;
                p_in != img_in + img_size - img_width;
                p_in += img_width, p_out += img_width) {
            *p_out = *p_in;
        }

        // Last row
        for (unsigned char *p_in = img_in + img_size - img_width, *p_out = img_out + img_size - img_width;
                p_in != img_in + img_size;
                p_in += 1, p_out += 1) {
            *p_out = *p_in;
        }

        // Last column
        for (unsigned char *p_in = img_in + img_width * 2 - 1, *p_out = img_out + img_width * 2 - 1;
                p_in != img_in + img_size - img_width - 1;
                p_in += img_width, p_out += img_width) {
            *p_out = *p_in;
        }

        // Convolution
        for (unsigned char *p_in = img_in + img_width + 1, *p_out = img_out + img_width + 1;
                p_in != img_in + img_size - img_width - 1;
                p_in += 1, p_out += 1) {
            float sum = 0.0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    sum += kernel[i + 1][j + 1] * *(p_in + i * img_width + j);
                }
            }
            *p_out = (unsigned char) sum;
        }


        // Measurement of the time it takes each process to calculate
        memset(&t1, 0, sizeof(struct timeval));
        memset(&t2, 0, sizeof(struct timeval));
        gettimeofday(&t1, NULL);

        gettimeofday(&t2, NULL);


        // Regrouping of the partial results of each process
        memset(&t1, 0, sizeof(struct timeval));
        memset(&t2, 0, sizeof(struct timeval));
        gettimeofday(&t1, NULL);
        
        gettimeofday(&t2, NULL);
*/
    }
    
    if (myrank == MASTER_RANK) {

        stbi_write_jpg(f_out, img_width, img_height, 1, img_out, 100);
        
        stbi_image_free(img_in);
        stbi_image_free(img_out);

        MPI_Type_free(&col_type);
    }

    for (i = 0; i < K_ROWS; i++) {
        free(kernel[i]);
    }
    free(kernel);

    // Termination of MPI communication
    MPI_Finalize();

    exit(EXIT_SUCCESS);
}