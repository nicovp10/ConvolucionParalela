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
#define MAX_NAME_LENGTH 50
#define MAX_COLUMN_LENGTH 10000

#define ITER 1


int main(int argc, char *argv[]) {
    int myrank, mpi_comm_size, C, a, i, j, k;
    //struct timeval t1, t2;

    const char *f_in;
    char *f_out;
    unsigned char *img_in, *img_out;
    int img_width, img_height;
    size_t img_size;

    MPI_Datatype block_type;
    int total_blocks, processed_blocks, rest_columns, num_cycles, rest_processes, num_proc, offset;
    unsigned char *p_in, *p_out;

    MPI_Status status;
    unsigned char *block;
    int block_size;

    int env = 0, recv = 0;

    
    // Initialization of MPI communication
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
    

    // Check the arguments
    f_out = NULL;
    switch (argc) {
        case 2:
            C = atoi(argv[1]);
            f_in = "images/input.jpg";
            f_out = (char *) malloc(MAX_NAME_LENGTH * sizeof(char));
            if (f_out == NULL) {
                printf("ERROR: unable to allocate memory for the output image filename.\n\n");
                exit(EXIT_FAILURE);
            }
            sprintf(f_out, "images/output_%d.jpg", mpi_comm_size);
            break;
        case 3:
            C = atoi(argv[1]);
            f_in = argv[2];
            f_out = (char *) malloc(MAX_NAME_LENGTH * sizeof(char));
            if (f_out == NULL) {
                printf("ERROR: unable to allocate memory for the output image filename.\n\n");
                exit(EXIT_FAILURE);
            }
            sprintf(f_out, "images/output_%d.jpg", mpi_comm_size);
            break;
        case 4:
            C = atoi(argv[1]);
            f_in = argv[2];
            f_out = argv[3];
            break;
        default:
            printf("ERROR: incorrect number of parameters.\n\tUse ./program C [INPUT_IMAGE_FILENAME] [OUTPUT_IMAGE_FILENAME].\n\n");
            exit(EXIT_FAILURE);
    }


    // Allocate a contiguous block of memory for the kernel
    float (*kernel)[K_COLUMNS] = malloc(K_ROWS * sizeof(*kernel));
    if (kernel == NULL) {
        printf("ERROR: unable to allocate memory for the kernel.\n\n");
        exit(EXIT_FAILURE);
    }


    // Do ITER iterations instead of just one to have multiple reference values
    for (a = 1; a <= ITER; a++) {

        // The master process read the input image and define the kernel
        if (myrank == MASTER_RANK) {
            printf("iteration: %d\n", a);

            // Load the input image
            img_in = stbi_load(f_in, &img_width, &img_height, NULL, 1);
            if (img_in == NULL) {
                printf("ERROR: cannot loading the image.\n\n");
                exit(EXIT_FAILURE);
            }
            img_size = img_width * img_height;

            // Allocate memory for the output image
            img_out = (unsigned char *) malloc(img_size * sizeof(unsigned char));
            if (img_out == NULL) {
                printf("ERROR: unable to allocate memory for the output image.\n\n");
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

            // Determine the number of blocks to scatter the image columns and the rest
            total_blocks = img_width / C;
            rest_columns = img_width % C; 
            if (rest_columns > 0) {
                total_blocks++;
            }

            // Determine the number of cycles that each process have to do and the rest
            num_cycles = total_blocks / mpi_comm_size;
            rest_processes = total_blocks % mpi_comm_size;


            // Send the corresponding colums to each process 
            if (mpi_comm_size > 1) {
                processed_blocks = 1;
                num_proc = 1;
                while (processed_blocks < total_blocks - 1) {
                    MPI_Send(img_in + processed_blocks * C, 1, block_type, num_proc, 0, MPI_COMM_WORLD);
                    processed_blocks++;
                    env++;
                    printf("[Proceso %d] Envíos: %d\n", myrank, env);
                    num_proc++;
                    num_proc %= mpi_comm_size;
                    // If the block is the last one, the num_proc must not be modified
                    if (num_proc == 0 && processed_blocks < total_blocks - 1) {
                        num_proc++;
                        processed_blocks++;
                    }
                }

                // Last block
                //      If the master process doesn't have to convolve it, it will be send
                if (num_proc != MASTER_RANK) {
                    if (rest_columns > 0) {
                        MPI_Type_free(&block_type);
                        MPI_Type_vector(img_height, rest_columns, img_width, MPI_UNSIGNED_CHAR, &block_type);
                        MPI_Type_commit(&block_type);
                    }
                    MPI_Send(img_in + (total_blocks - 1) * C, 1, block_type, num_proc, 0, MPI_COMM_WORLD);
                    env++;
                    printf("[Proceso %d] Envíos: %d\n", myrank, env);
                }

                MPI_Type_free(&block_type);
            }
        }


        // Reception of the kernel, the number of cycles and the rest of processes from the master process
        if (mpi_comm_size > 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(&(kernel[0][0]), K_ROWS*K_COLUMNS, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
            MPI_Bcast(&num_cycles, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
            MPI_Bcast(&rest_processes, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
        }


        if (myrank == MASTER_RANK) {
            // Convolution of the master process
            processed_blocks = 0;
            while (processed_blocks < total_blocks - 1) {
                offset = processed_blocks * C;
                for (j = 0; j < C; j++) {
                    for (p_in = img_in + offset + j, p_out = img_out + offset + j;
                        p_in != img_in + img_size + offset + j;
                        p_in += img_width, p_out += img_width) {
                            *p_out = *p_in;
                    }
                }
                processed_blocks += mpi_comm_size;
            }

            if (processed_blocks == total_blocks) {
                if (rest_columns == 0) {
                    rest_columns = C;
                }
                offset = (total_blocks - 1) * C;
                for (j = 0; j < rest_columns; j++) {
                    for (p_in = img_in + offset + j, p_out = img_out + offset + j;
                        p_in != img_in + img_size + offset + j;
                        p_in += img_width, p_out += img_width) {
                            *p_out = *p_in;
                    }
                }
            }
            

            // Reception of the blocks from the slave processes
            if (mpi_comm_size > 1) {
                block_size = C * img_height;
                block = (unsigned char *) malloc(block_size * sizeof(unsigned char));

                for (i = 1; i < mpi_comm_size; i++) {
                    for (j = 0; j < num_cycles; j++) {
                        offset = (i + j * mpi_comm_size) * C;
                        MPI_Recv(block, block_size, MPI_UNSIGNED_CHAR, i, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        recv++;
                        printf("[Proceso %d] Recepcións: %d\n", myrank, recv);
                        for (k = 0; k < C; k++) {
                            for (p_in = block + k, p_out = img_out + offset + k;
                                p_in != block + block_size + k;
                                p_in += C, p_out += img_width) {
                                    *p_out = *p_in;
                            }
                        }
                    }
                }

                // Extra cycle
                if (rest_processes > 1) {
                    for (i = 1; i < rest_processes - 1; i++) {
                        offset = (i + num_cycles * mpi_comm_size) * C;
                        MPI_Recv(block, block_size, MPI_UNSIGNED_CHAR, i, num_cycles, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        recv++;
                        printf("[Proceso %d] Recepcións: %d\n", myrank, recv);
                        for (j = 0; j < C; j++) {
                            for (p_in = block + j, p_out = img_out + offset + j;
                                p_in != block + block_size + j;
                                p_in += C, p_out += img_width) {
                                    *p_out = *p_in;
                            }
                        }
                    }

                    // Last process of the extra cycle
                    MPI_Recv(block, block_size, MPI_UNSIGNED_CHAR, rest_processes - 1, num_cycles, MPI_COMM_WORLD, &status);
                    MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &block_size);
                    recv++;
                    printf("[Proceso %d] Recepcións: %d\n", myrank, recv);
                    offset = ((rest_processes - 1) + num_cycles * mpi_comm_size) * C;
                    for (i = 0; i < rest_columns; i++) {
                        for (p_in = block + i, p_out = img_out + offset + i;
                            p_in != block + block_size + i;
                            p_in += rest_columns, p_out += img_width) {
                                *p_out = *p_in;
                        }
                    }
                }
                free(block);
            }

            stbi_write_jpg(f_out, img_width, img_height, 1, img_out, 100);
            
            stbi_image_free(img_in);
            stbi_image_free(img_out);
        } else {
            block = (unsigned char *) malloc(C * MAX_COLUMN_LENGTH * sizeof(unsigned char));
            
            MPI_Recv(block, C * MAX_COLUMN_LENGTH, MPI_UNSIGNED_CHAR, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            recv++;
            printf("[Proceso %d] Recepcións: %d\n", myrank, recv);
            MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &block_size);
            block = (unsigned char *) realloc(block, block_size * sizeof(unsigned char));
            // CONVOLUTION OPERATION
            MPI_Send(block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, 0, MPI_COMM_WORLD);
            env++;
            printf("[Proceso %d] Envíos: %d\n", myrank, env);

            for (i = 1; i < num_cycles; i++) {
                MPI_Recv(block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recv++;
                printf("[Proceso %d] Recepcións: %d\n", myrank, recv);
                // CONVOLUTION OPERATION
                MPI_Send(block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, i, MPI_COMM_WORLD);
                env++;
                printf("[Proceso %d] Envíos: %d\n", myrank, env);
            }
            
            if (rest_processes > 1 && myrank < rest_processes) {
                MPI_Recv(block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &status);                    
                recv++;
                printf("[Proceso %d] Recepcións: %d\n", myrank, recv);
                if (myrank == rest_processes - 1) {
                    MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &block_size);
                }
                // CONVOLUTION OPERATION
                MPI_Send(block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, num_cycles, MPI_COMM_WORLD);
                env++;
                printf("[Proceso %d] Envíos: %d\n", myrank, env);
            }
            printf("[Proceso %d] Fin\n", myrank);
            free(block);
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

    }

    free(kernel);
    if (f_out != NULL) {
        free(f_out);
    }
    
    // Termination of MPI communication
    MPI_Finalize();

    exit(EXIT_SUCCESS);
}