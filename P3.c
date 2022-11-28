#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include "mpi.h"
 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define ITER 1
#define C 1


int main(int argc, char *argv[]) {
    int myrank, mpi_size, a;
    struct timeval t1, t2;
    const char *f_in, *f_out;
    unsigned char *img_in, *img_out;
    int img_width, img_height;
    MPI_Datatype col_type;

    
    // Initialization of MPI communication
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


    if (argc == 3) {
        f_in = argv[1];
        f_out = argv[2];
    } else {
        f_in = "input.jpg";
        f_out = "output.jpg";
    }

            
    img_in = stbi_load(f_in, &img_width, &img_height, NULL, 1);
    if(img_in == NULL) {
        printf("Error in loading the image\n");
        exit(EXIT_FAILURE);
    }

    size_t img_size = img_width * img_height;

    for (a = 0; a < ITER; a++) {
        if (myrank == 0) {
            printf("Iteración: 1\n");

            img_out = malloc(img_size);
            if(img_out == NULL) {
                printf("Unable to allocate memory for the convolved image.\n");
                exit(EXIT_FAILURE);
            }

            // Convolution kernel for shaping
            float kernel[3][3] = {
                {0, -1, 0},
                {-1, 6, -1},
                {0, -1, 0}};
        
            // Define the MPI Datatype column
            MPI_Type_vector(img_height, 1, img_width, MPI_UNSIGNED_CHAR, &col_type);
            MPI_Type_commit(&col_type);

            printf("Columna lida polo proceso máster:\n\t");
            for (unsigned char *p = img_in; p != img_in + img_size - img_width; p += img_width) {
                printf("%d ", *p);
            }
            printf("\n");
            MPI_Send(img_in, 1, col_type, 1, 0, MPI_COMM_WORLD);
        } else {
            unsigned char col[img_height];
            MPI_Recv(col, img_height, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Columna lida polo proceso esclavo:\n\t");
            for (int i = 0; i < img_height; i++) {
                printf("%d ", col[i]);
            }
            printf("\n");
        }

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
        if (myrank == 0) {


            stbi_write_jpg(f_out, img_width, img_height, 1, img_out, 100);
            
            stbi_image_free(img_in);
            stbi_image_free(img_out);
            MPI_Type_free(&col_type);
        }
    }
    

    // Termination of MPI communication
    MPI_Finalize();

    exit(EXIT_SUCCESS);
}