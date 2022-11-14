#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <pmmintrin.h>
#include <sys/types.h>
#include <sys/time.h>
#include "mpi.h"


#define LINESIZE 64
#define ITER 5
#define C 1


int main(int argc, char *argv[]) {

    int myrank, size, a;
    long i, j; 
    struct timeval t1, t2;

    
    // Initialization of MPI communication
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    for (a = 0; a < ITER; a++){

        // Get the image and kernel
        if(myrank == 0){
            printf("Iter%d\n", a);

            FILE *image_file = fopen("input.bmp", "r");
            if (image_file == NULL) {
                printf("ERRO: non se puido abrir a imaxe.\n");
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }

            // Read the 54 byte header from the imaxe
            unsigned char imaxe_header[54];
            for (i = 0; i < 54; i++) {
                imaxe_header[i] = getc(image_file);
            }

            // Extract image height and width the image header
            long image_height = *(int*)&imaxe_header[18];
            long image_width = *(int*)&imaxe_header[22];

            // Create a matrix for the image data
            unsigned char** image_matrix=(unsigned char**) _mm_malloc(image_height * sizeof(unsigned char*), LINESIZE);
            for (i = 0; i < image_height; i++) {
                image_matrix[i]=(unsigned char*) malloc(image_width * sizeof(unsigned char));
            }

            // Split the global matrix into parcial matrix for each process


            // Put the image data into the matrix
            for (i = 0; i < image_height; i++) {
                for (j = 0; j < image_width; j++) {
                    image_matrix[i][j] = getc(fIn);
                }
            }
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
        
        
        if (myrank == 0) {


            // Frees the matrix memory
            for (i = 0; i < image_height; i++) {
                free(image_matrix[i]);
            }
            _mm_free(image_matrix);
            
            // Close the input imaxe
            fclose(image_file);
        }
    }


    // Termination of MPI communication
    MPI_Finalize();

    exit(EXIT_SUCCESS);
}