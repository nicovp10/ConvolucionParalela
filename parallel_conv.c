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
#define MAX_COLUMN_LENGTH 10000

#define ITER 1
#define C 8


int main(int argc, char *argv[]) {
    int myrank, mpi_comm_size, a, i, j;
    struct timeval t1, t2;
    const char *f_in;
    char *f_out;
    unsigned char *img_in, *img_out;
    int img_width, img_height;
    size_t img_size;
    MPI_Datatype block_type;
    float** kernel;
    float num_cycles;
    MPI_Status status;
    unsigned char *block;
    int block_size;

    int env = 0, recv = 0;

    
    // Initialization of MPI communication
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
    

    for (a = 1; a <= ITER; a++) {
        // Allocate memory for the kernel
        kernel = malloc(K_ROWS * sizeof(float*));
        for (i = 0; i < K_ROWS; i++) {
            kernel[i] = malloc(K_COLUMNS * sizeof(float));
        }
        

        if (myrank == MASTER_RANK) {
            int total_blocks, send_blocks, rest_columns, num_proc;


            printf("Iteration: %d\n", a);


            // Set image filenames
            switch (argc) {
                case 2:
                    f_in = argv[1];
                    sprintf(f_out, "images/output_%d.jpg", mpi_comm_size);
                    break;
                case 3:
                    f_in = argv[1];
                    f_out = argv[2];
                    break;
                default:
                    f_in = "images/input.jpg";
                    sprintf(f_out, "images/output_%d.jpg", mpi_comm_size);
                    break;
            }

            // Load the input image
            img_in = stbi_load(f_in, &img_width, &img_height, NULL, 1);
            if(img_in == NULL) {
                printf("Error in loading the image.\n");
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
        

            // Define the MPI Datatype
            MPI_Type_vector(img_height, C, img_width, MPI_UNSIGNED_CHAR, &block_type);
            MPI_Type_commit(&block_type);

            // Determine the number of iterations to scatter the image columns
            total_blocks = img_width / C;
            rest_columns = img_width % C; 
            if (rest_columns != 0) {
                total_blocks++;
            }
            num_cycles = total_blocks / (mpi_comm_size - 1);
            if (total_blocks % (mpi_comm_size - 1) != 0) {
                num_cycles++;
            }

            // Send the corresponding colums to each process 
            send_blocks = 0;
            num_proc = 1;
            while (send_blocks < total_blocks - 1) {
                MPI_Send(img_in + send_blocks * C, 1, block_type, num_proc, 0, MPI_COMM_WORLD);
                send_blocks++;
                env++;
                printf("[Proceso %d] Envíos: %d\n", myrank, env);
                //num_proc++;
                //num_proc %= mpi_comm_size;
            }

            if (rest_columns != 0) {
                MPI_Type_free(&block_type);
                MPI_Type_vector(img_height, rest_columns, img_width, MPI_UNSIGNED_CHAR, &block_type);
                MPI_Type_commit(&block_type);
            }
            MPI_Send(img_in + (total_blocks - 1) * C, 1, block_type, num_proc, 0, MPI_COMM_WORLD);
            env++;
            printf("[Proceso %d] Envíos: %d\n", myrank, env);
        }

        // Receive the kernel and the number of cycles from the master process
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&(kernel[0][0]), K_ROWS*K_COLUMNS, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Bcast(&num_cycles, 1, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

        if (myrank != MASTER_RANK) {
            block = malloc(C * MAX_COLUMN_LENGTH * sizeof(unsigned char*));
            block_size = 0;
            
            MPI_Recv(block, C * MAX_COLUMN_LENGTH, MPI_UNSIGNED_CHAR, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            recv++;
            printf("[Proceso %d] Recepcións: %d\n", myrank, recv);
            MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &block_size);
            block = realloc(block, block_size * sizeof(unsigned char*));
            printf("block_size=%d\n", block_size);
            printf("RECIBIDO\n");
            for (unsigned char *p_in = block; p_in != block + block_size; p_in += C) {
                printf("%d ", *p_in);
            }
            printf("\n\n");
            // CONVOLUTION OPERATION
            MPI_Send(block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, 0, MPI_COMM_WORLD);
            env++;
            printf("[Proceso %d] Envíos: %d\n", myrank, env);

            for (i = 1; i < num_cycles - 1; i++) {
                MPI_Recv(block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recv++;
                printf("[Proceso %d] Recepcións: %d\n", myrank, recv);
                // CONVOLUTION OPERATION
                MPI_Send(block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, 0, MPI_COMM_WORLD);
                env++;
                printf("[Proceso %d] Envíos: %d\n", myrank, env);
            }
            /*
            int rest_cycles = num_cycles - (int) num_cycles;
            int num_processes = rest_cycles * mpi_comm_size;
            if (num_processes != 0) {
                for (i = 0; i < num_processes - 1; i++) {
                    MPI_Recv(block, MAX_COLUMN_LENGTH * C, MPI_UNSIGNED_CHAR, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // CONVOLUTION OPERATION
                    MPI_Send(block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, 0, MPI_COMM_WORLD);
                }
            }*/

            MPI_Recv(block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            recv++;
            printf("[Proceso %d] Recepcións: %d\n", myrank, recv);
            MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &block_size);
            // CONVOLUTION OPERATION
            MPI_Send(block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, 0, MPI_COMM_WORLD);
            env++;
            printf("[Proceso %d] Envíos: %d\n", myrank, env);
            printf("[Proceso %d] Fin\n", myrank);
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
    
        if (myrank == MASTER_RANK) {
            int offset, last_columns;
            block_size = C * img_height;
            block = malloc(block_size * sizeof(unsigned char*));
            printf("num_cycles = %f\n", num_cycles);
            for (i = 0; i < num_cycles - 1; i++) {
                offset = C * i;
                MPI_Recv(block, block_size, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recv++;
                printf("[Proceso %d] Recepcións: %d\n", myrank, recv);
                for (j = 0; j < C; j++) {
                    for (unsigned char *p_in = block + j, *p_out = img_out + offset + j;
                        p_in != block + block_size + j;
                        p_in += C, p_out += img_width) {
                            *p_out = *p_in;
                    }
                }
            }
            
            MPI_Recv(block, block_size, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            recv++;
            printf("[Proceso %d] Recepcións: %d\n", myrank, recv);
            MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &block_size);
            last_columns = block_size / img_height;
            offset = C * (num_cycles - 1);
            for (i = 0; i < last_columns; i++) {
                for (unsigned char *p_in = block + i, *p_out = img_out + offset + i;
                    p_in != block + block_size + i;
                    p_in += last_columns, p_out += img_width) {
                        *p_out = *p_in;
                }
            }

            stbi_write_jpg(f_out, img_width, img_height, 1, img_out, 100);
            
            stbi_image_free(img_in);
            stbi_image_free(img_out);

            MPI_Type_free(&block_type);
        }

        for (i = 0; i < K_ROWS; i++) {
            free(kernel[i]);
        }
        free(kernel);
        free(block);
    }
    
    // Termination of MPI communication
    MPI_Finalize();

    exit(EXIT_SUCCESS);
}