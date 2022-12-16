#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define MASTER_RANK 0

#define MAX_FILENAME_LENGTH 50
#define MAX_COLUMN_LENGTH 10000

#define K_ROWS 3
#define K_COLUMNS 3


int main(int argc, char *argv[]) {
    int myrank, mpi_comm_size, C, i, j, k, x, y;
    struct timeval t1, t2;

    const char *f_in;
    char *f_out;
    unsigned char *img_in, *img_out;
    int img_width, img_height;
    size_t img_size;

    MPI_Datatype block_type;
    int total_blocks, processed_blocks, rest_columns, num_cycles, rest_processes, num_proc, offset;
    unsigned char *p_in, *p_out;
    float auxSum;

    MPI_Status status;
    unsigned char *send_block, *recv_block;
    int block_size;
    int block_column_length;

    
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
            f_out = (char *) malloc(MAX_FILENAME_LENGTH * sizeof(char));
            if (f_out == NULL) {
                printf("ERROR: unable to allocate memory for the output image filename.\n\n");
                exit(EXIT_FAILURE);
            }
            sprintf(f_out, "images/output_%d_%d.jpg", C, mpi_comm_size);
            break;
        case 3:
            C = atoi(argv[1]);
            f_in = argv[2];
            f_out = (char *) malloc(MAX_FILENAME_LENGTH * sizeof(char));
            if (f_out == NULL) {
                printf("ERROR: unable to allocate memory for the output image filename.\n\n");
                exit(EXIT_FAILURE);
            }
            sprintf(f_out, "images/output_%d_%d.jpg", C, mpi_comm_size);
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


    // The master process does various tasks
    if (myrank == MASTER_RANK) {
        printf("Init\n");

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
        //      C + 2 because of the convolution of the first and last columns of the block
        MPI_Type_vector(img_height, C + 2, img_width, MPI_UNSIGNED_CHAR, &block_type);
        MPI_Type_commit(&block_type);

        // Determine the number of blocks to scatter the image columns and the rest
        total_blocks = img_width / C;
        rest_columns = img_width % C;

        // Determine the number of cycles that each process have to do and the rest
        num_cycles = total_blocks / mpi_comm_size;
        rest_processes = total_blocks % mpi_comm_size;

        // Send the corresponding blocks to each process 
        if (mpi_comm_size > 1) {
            memset(&t1, 0, sizeof(struct timeval));
            memset(&t2, 0, sizeof(struct timeval));
            gettimeofday(&t1, NULL);

            processed_blocks = 1;
            num_proc = 1;
            while (processed_blocks < total_blocks) {
                // Substract 1 from the offset to have the information from the column before the first column of the block
                MPI_Send(img_in + processed_blocks * C - 1, 1, block_type, num_proc, 0, MPI_COMM_WORLD);
                processed_blocks++;
                num_proc++;
                num_proc %= mpi_comm_size;
                // If the block is the last one, the num_proc must not be modified
                if (num_proc == 0 && processed_blocks < total_blocks) {
                    num_proc++;
                    processed_blocks++;
                }
            }

            // Extra block
            // If the master process doesn't have to convolve it, it will be send
            if (rest_columns > 0 && num_proc != MASTER_RANK) {
                MPI_Type_free(&block_type);
                // rest_colums + 1 because of the first column of the last block of the image
                MPI_Type_vector(img_height, rest_columns + 1, img_width, MPI_UNSIGNED_CHAR, &block_type);
                MPI_Type_commit(&block_type);
                MPI_Send(img_in + total_blocks * C - 1, 1, block_type, num_proc, 0, MPI_COMM_WORLD);
            }

            MPI_Type_free(&block_type);

            gettimeofday(&t2, NULL);
            printf("Sends: %lf\n", (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6));
        }
    }
    

    // Reception of the kernel, the number of cycles and the rest of processes from the master process
    if (mpi_comm_size > 1) {
        // Barrier to wait for master process
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (myrank == MASTER_RANK) {
            memset(&t1, 0, sizeof(struct timeval));
            memset(&t2, 0, sizeof(struct timeval));
            gettimeofday(&t1, NULL);
        }
        
        MPI_Bcast(&(kernel[0][0]), K_ROWS*K_COLUMNS, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Bcast(&num_cycles, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Bcast(&rest_processes, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
        MPI_Bcast(&rest_columns, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);

        if (myrank == MASTER_RANK) {
            gettimeofday(&t2, NULL);
            printf("Bcasts: %lf\n", (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6));
        }
    }


    if (myrank == MASTER_RANK) {
        memset(&t1, 0, sizeof(struct timeval));
        memset(&t2, 0, sizeof(struct timeval));
        gettimeofday(&t1, NULL);

        // CONVOLUTION OF THE MASTER PROCESS
        // First block of the image
        // First column of the first block of the image
        // First element of the column
        p_in = img_in;
        p_out = img_out;
        auxSum = 0.0;
        for (x = 0; x <= 1; x++) {
            for (y = 0; y <= 1; y++) {
                auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
            }
        }
        *p_out = (unsigned char) auxSum;

        // Middle elements of the column
        for (p_in = img_in + img_width, p_out = img_out + img_width;
            p_in != img_in + img_size - img_width;
            p_in += img_width, p_out += img_width) {
                auxSum = 0.0;
                for (x = -1; x <= 1; x++) {
                    for (y = 0; y <= 1; y++) {
                        auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                    }
                }
                *p_out = (unsigned char) auxSum;
        }

        // Last element of the column
        p_in = img_in + img_size - img_width;
        p_out = img_out + img_size - img_width;
        auxSum = 0.0;
        for (x = -1; x <= 0; x++) {
            for (y = 0; y <= 1; y++) {
                auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
            }
        }
        *p_out = (unsigned char) auxSum;

        // Rest of the columns of the first block of the image
        for (i = 1; i < C; i++) {
            // First element of the column
            p_in = img_in + i;
            p_out = img_out + i;
            auxSum = 0.0;
            for (x = 0; x <= 1; x++) {
                for (y = -1; y <= 1; y++) {
                    auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                }
            }
            *p_out = (unsigned char) auxSum;

            // Middle elements of the column
            for (p_in = img_in + i + img_width, p_out = img_out + i + img_width;
                p_in != img_in + i + img_size - img_width;
                p_in += img_width, p_out += img_width) {
                    auxSum = 0.0;
                    for (x = -1; x <= 1; x++) {
                        for (y = -1; y <= 1; y++) {
                            auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                        }
                    }
                    *p_out = (unsigned char) auxSum;
            }

            // Last element of the column
            p_in = img_in + i + img_size - img_width;
            p_out = img_out + i + img_size - img_width;
            auxSum = 0.0;
            for (x = -1; x <= 0; x++) {
                for (y = -1; y <= 1; y++) {
                    auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                }
            }
            *p_out = (unsigned char) auxSum;
        }


        // Middle blocks
        processed_blocks = mpi_comm_size;
        while (processed_blocks < total_blocks) {
            offset = processed_blocks * C;
            for (i = 0; i < C; i++) {
                // First element of the column
                p_in = img_in + offset + i;
                p_out = img_out + offset + i;
                auxSum = 0.0;
                for (x = 0; x <= 1; x++) {
                    for (y = -1; y <= 1; y++) {
                        auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                    }
                }
                *p_out = (unsigned char) auxSum;

                // Middle elements of the column
                for (p_in = img_in + offset + i + img_width, p_out = img_out + offset + i + img_width;
                    p_in != img_in + offset + i + img_size - img_width;
                    p_in += img_width, p_out += img_width) {
                        auxSum = 0.0;
                        for (x = -1; x <= 1; x++) {
                            for (y = -1; y <= 1; y++) {
                                auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                            }
                        }
                        *p_out = (unsigned char) auxSum;
                }

                // Last element of the column
                p_in = img_in + offset + i + img_size - img_width;
                p_out = img_out + offset + i + img_size - img_width;
                auxSum = 0.0;
                for (x = -1; x <= 0; x++) {
                    for (y = -1; y <= 1; y++) {
                        auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                    }
                }
                *p_out = (unsigned char) auxSum;
            }
            processed_blocks += mpi_comm_size;
        }

        // Last block of the image
        if (processed_blocks == total_blocks && rest_columns > 0) {
            offset = total_blocks * C;

            // Firsts columns of the last block of the image
            for (i = 0; i < rest_columns - 1; i++) {
                // First element of the column
                p_in = img_in + offset + i;
                p_out = img_out + offset + i;
                auxSum = 0.0;
                for (x = 0; x <= 1; x++) {
                    for (y = -1; y <= 1; y++) {
                        auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                    }
                }
                *p_out = (unsigned char) auxSum;

                // Middle elements of the column
                for (p_in = img_in + offset + i + img_width, p_out = img_out + offset + i + img_width;
                    p_in != img_in + offset + i + img_size - img_width;
                    p_in += img_width, p_out += img_width) {
                        auxSum = 0.0;
                        for (x = -1; x <= 1; x++) {
                            for (y = -1; y <= 1; y++) {
                                auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                            }
                        }
                        *p_out = (unsigned char) auxSum;
                }

                // Last element of the column
                p_in = img_in + offset + i + img_size - img_width;
                p_out = img_out + offset + i + img_size - img_width;
                auxSum = 0.0;
                for (x = -1; x <= 0; x++) {
                    for (y = -1; y <= 1; y++) {
                        auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                    }
                }
                *p_out = (unsigned char) auxSum;
            }

            // Last column of the last block of the image
            // First element of the column
            offset = total_blocks * C + rest_columns - 1;
            p_in = img_in + offset;
            p_out = img_out + offset;
            auxSum = 0.0;
            for (x = 0; x <= 1; x++) {
                for (y = -1; y <= 0; y++) {
                    auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                }
            }
            *p_out = (unsigned char) auxSum;

            // Middle elements of the column
            for (p_in = img_in + offset + img_width, p_out = img_out + offset + img_width;
                p_in != img_in + offset + img_size - img_width;
                p_in += img_width, p_out += img_width) {
                    auxSum = 0.0;
                    for (x = -1; x <= 1; x++) {
                        for (y = -1; y <= 0; y++) {
                            auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                        }
                    }
                    *p_out = (unsigned char) auxSum;
            }

            // Last element of the column
            p_in = img_in + offset + img_size - img_width;
            p_out = img_out + offset + img_size - img_width;
            auxSum = 0.0;
            for (x = -1; x <= 0; x++) {
                for (y = -1; y <= 0; y++) {
                    auxSum += kernel[x + 1][y + 1] * *(p_in + x * img_width + y);
                }
            }
            *p_out = (unsigned char) auxSum;
        }

        gettimeofday(&t2, NULL);
        printf("Process: %d; Convolutions: %lf\n", myrank, (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6));


        // Reception of the blocks from the slave processes
        if (mpi_comm_size > 1) {
            memset(&t1, 0, sizeof(struct timeval));
            memset(&t2, 0, sizeof(struct timeval));
            gettimeofday(&t1, NULL);

            block_size = (C + 2) * img_height;
            recv_block = (unsigned char *) malloc(block_size * sizeof(unsigned char));

            for (i = 1; i < mpi_comm_size; i++) {
                for (j = 0; j < num_cycles; j++) {
                    offset = (i + j * mpi_comm_size) * C;
                    MPI_Recv(recv_block, block_size, MPI_UNSIGNED_CHAR, i, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (k = 0; k < C; k++) {
                        for (p_in = recv_block + k + 1, p_out = img_out + offset + k;
                            p_in != recv_block + block_size + k + 1;
                            p_in += C + 2, p_out += img_width) {
                                *p_out = *p_in;
                        }
                    }
                }
            }

            // Extra cycle
            if (rest_processes > 1) {
                for (i = 1; i < rest_processes; i++) {
                    offset = (i + num_cycles * mpi_comm_size) * C;
                    MPI_Recv(recv_block, block_size, MPI_UNSIGNED_CHAR, i, num_cycles, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (j = 0; j < C; j++) {
                        for (p_in = recv_block + j + 1, p_out = img_out + offset + j;
                            p_in != recv_block + block_size + j + 1;
                            p_in += C + 2, p_out += img_width) {
                                *p_out = *p_in;
                        }
                    }
                }

                // Process with the last block of the image
                if (rest_columns > 0) {
                    MPI_Recv(recv_block, block_size, MPI_UNSIGNED_CHAR, rest_processes, num_cycles, MPI_COMM_WORLD, &status);
                    MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &block_size);
                    offset = (rest_processes + num_cycles * mpi_comm_size) * C;
                    for (i = 0; i < rest_columns; i++) {
                        for (p_in = recv_block + i + 1, p_out = img_out + offset + i;
                            p_in != recv_block + block_size + i + 1;
                            p_in += rest_columns + 1, p_out += img_width) {
                                *p_out = *p_in;
                        }
                    }
                }
            }
            free(recv_block);

            gettimeofday(&t2, NULL);
            printf("Recvs: %lf\n", (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6));
        }

        stbi_write_jpg(f_out, img_width, img_height, 1, img_out, 100);
        
        stbi_image_free(img_in);
        stbi_image_free(img_out);
    } else {
        memset(&t1, 0, sizeof(struct timeval));
        memset(&t2, 0, sizeof(struct timeval));
        gettimeofday(&t1, NULL);

        recv_block = (unsigned char *) malloc((C + 2) * MAX_COLUMN_LENGTH * sizeof(unsigned char));
        if (recv_block == NULL) {
            printf("ERROR: unable to allocate memory for recv_block.\n\n");
            exit(EXIT_FAILURE);
        }
        
        MPI_Recv(recv_block, (C + 2) * MAX_COLUMN_LENGTH, MPI_UNSIGNED_CHAR, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &block_size);
        block_column_length = block_size / (C + 2);
        recv_block = (unsigned char *) realloc(recv_block, block_size * sizeof(unsigned char));
        if (recv_block == NULL) {
            printf("ERROR: unable to reallocate memory for recv_block.\n\n");
            exit(EXIT_FAILURE);
        }

        send_block = (unsigned char *) malloc(block_size * sizeof(unsigned char));
        if (send_block == NULL) {
            printf("ERROR: unable to allocate memory for send_block.\n\n");
            exit(EXIT_FAILURE);
        }

        // Aux columns
        for (p_in = recv_block, p_out = send_block;
            p_in != recv_block + block_size;
            p_in += C + 2, p_out += C + 2) {
                *p_out = *p_in;
        }
        for (p_in = recv_block + C + 1, p_out = send_block + C + 1;
            p_in != recv_block + block_size + C + 1;
            p_in += C + 2, p_out += C + 2) {
                *p_out = *p_in;
        }
        
        // Convolution
        for (i = 1; i <= C; i++) {
            // First element of the column
            p_in = recv_block + i;
            p_out = send_block + i;
            auxSum = 0.0;
            for (x = 0; x <= 1; x++) {
                for (y = -1; y <= 1; y++) {
                    auxSum += kernel[x + 1][y + 1] * *(p_in + x * (C + 2) + y);
                }
            }
            *p_out = (unsigned char) auxSum;

            // Middle elements of the column
            for (p_in = recv_block + i + C + 2, p_out = send_block + i + C + 2;
                p_in != recv_block + i + block_size - C - 2;
                p_in += C + 2, p_out += C + 2) {
                    auxSum = 0.0;
                    for (x = -1; x <= 1; x++) {
                        for (y = -1; y <= 1; y++) {
                            auxSum += kernel[x + 1][y + 1] * *(p_in + x * (C + 2) + y);
                        }
                    }
                    *p_out = (unsigned char) auxSum;
            }
            
            // Last element of the column
            p_in = recv_block + i + block_size - C - 2;
            p_out = send_block + i + block_size - C - 2;
            auxSum = 0.0;
            for (x = -1; x <= 0; x++) {
                for (y = -1; y <= 1; y++) {
                    auxSum += kernel[x + 1][y + 1] * *(p_in + x * (C + 2) + y);
                }
            }
            *p_out = (unsigned char) auxSum;
        }
        MPI_Send(send_block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, 0, MPI_COMM_WORLD);

        for (i = 1; i < num_cycles; i++) {
            MPI_Recv(recv_block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Convolution
            for (j = 1; j <= C; j++) {
                // First element of the column
                p_in = recv_block + j;
                p_out = send_block + j;
                auxSum = 0.0;
                for (x = 0; x <= 1; x++) {
                    for (y = -1; y <= 1; y++) {
                        auxSum += kernel[x + 1][y + 1] * *(p_in + x * (C + 2) + y);
                    }
                }
                *p_out = (unsigned char) auxSum;

                // Middle elements of the column
                for (p_in = recv_block + j + C + 2, p_out = send_block + j + C + 2;
                    p_in != recv_block + j + block_size - C - 2;
                    p_in += C + 2, p_out += C + 2) {
                        auxSum = 0.0;
                        for (x = -1; x <= 1; x++) {
                            for (y = -1; y <= 1; y++) {
                                auxSum += kernel[x + 1][y + 1] * *(p_in + x * (C + 2) + y);
                            }
                        }
                        *p_out = (unsigned char) auxSum;
                }

                // Last element of the column
                p_in = recv_block + j + block_size - C - 2;
                p_out = send_block + j + block_size - C - 2;
                auxSum = 0.0;
                for (x = -1; x <= 0; x++) {
                    for (y = -1; y <= 1; y++) {
                        auxSum += kernel[x + 1][y + 1] * *(p_in + x * (C + 2) + y);
                    }
                }
                *p_out = (unsigned char) auxSum;
            }
            MPI_Send(send_block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, i, MPI_COMM_WORLD);
        }
        
        // If the current process is one of those that have to do an extra cycle
        if (rest_processes > 1 && myrank < rest_processes) {
            MPI_Recv(recv_block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Convolution
            for (j = 1; j <= C; j++) {
                // First element of the column
                p_in = recv_block + j;
                p_out = send_block + j;
                auxSum = 0.0;
                for (x = 0; x <= 1; x++) {
                    for (y = -1; y <= 1; y++) {
                        auxSum += kernel[x + 1][y + 1] * *(p_in + x * (C + 2) + y);
                    }
                }
                *p_out = (unsigned char) auxSum;

                // Middle elements of the column
                for (p_in = recv_block + j + C + 2, p_out = send_block + j + C + 2;
                    p_in != recv_block + j + block_size - C - 2;
                    p_in += C + 2, p_out += C + 2) {
                        auxSum = 0.0;
                        for (x = -1; x <= 1; x++) {
                            for (y = -1; y <= 1; y++) {
                                auxSum += kernel[x + 1][y + 1] * *(p_in + x * (C + 2) + y);
                            }
                        }
                        *p_out = (unsigned char) auxSum;
                }

                // Last element of the column
                p_in = recv_block + j + block_size - C - 2;
                p_out = send_block + j + block_size - C - 2;
                auxSum = 0.0;
                for (x = -1; x <= 0; x++) {
                    for (y = -1; y <= 1; y++) {
                        auxSum += kernel[x + 1][y + 1] * *(p_in + x * (C + 2) + y);
                    }
                }
                *p_out = (unsigned char) auxSum;
            }
            MPI_Send(send_block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, num_cycles, MPI_COMM_WORLD);
        }

        // If the current process is the one that have to convolve the last block of the image
        if (rest_columns > 0 && myrank == rest_processes) {
            MPI_Recv(recv_block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &block_size);
            rest_columns = block_size / block_column_length;

            // Convolution of the last block of the image
            // First columns of the last block of the image
            for (i = 1; i < rest_columns - 1; i++) {
                // First element of the column
                p_in = recv_block + i;
                p_out = send_block + i;
                auxSum = 0.0;
                for (x = 0; x <= 1; x++) {
                    for (y = -1; y <= 1; y++) {
                        auxSum += kernel[x + 1][y + 1] * *(p_in + x * rest_columns + y);
                    }
                }
                *p_out = (unsigned char) auxSum;

                // Middle elements of the column
                for (p_in = recv_block + i + rest_columns, p_out = send_block + i + rest_columns;
                    p_in != recv_block + i + block_size - rest_columns;
                    p_in += rest_columns, p_out += rest_columns) {
                        auxSum = 0.0;
                        for (x = -1; x <= 1; x++) {
                            for (y = -1; y <= 1; y++) {
                                auxSum += kernel[x + 1][y + 1] * *(p_in + x * rest_columns + y);
                            }
                        }
                        *p_out = (unsigned char) auxSum;
                }

                // Last element of the column
                p_in = recv_block + i + block_size - rest_columns;
                p_out = send_block + i + block_size - rest_columns;
                auxSum = 0.0;
                for (x = -1; x <= 0; x++) {
                    for (y = -1; y <= 1; y++) {
                        auxSum += kernel[x + 1][y + 1] * *(p_in + x * rest_columns + y);
                    }
                }
                *p_out = (unsigned char) auxSum;
            }

            // Last column of the last block of the image
            // First element of the column
            offset = rest_columns - 1;
            p_in = recv_block + offset;
            p_out = send_block + offset;
            auxSum = 0.0;
            for (x = 0; x <= 1; x++) {
                for (y = -1; y <= 1; y++) {
                    auxSum += kernel[x + 1][y + 1] * *(p_in + x * rest_columns + y);
                }
            }
            *p_out = (unsigned char) auxSum;

            // Middle elements of the column
            for (p_in = recv_block + offset + rest_columns, p_out = send_block + offset + rest_columns;
                p_in != recv_block + offset + block_size - rest_columns;
                p_in += rest_columns, p_out += rest_columns) {
                    auxSum = 0.0;
                    for (x = -1; x <= 1; x++) {
                        for (y = -1; y <= 1; y++) {
                            auxSum += kernel[x + 1][y + 1] * *(p_in + x * rest_columns + y);
                        }
                    }
                    *p_out = (unsigned char) auxSum;
            }

            // Last element of the column
            p_in = recv_block + offset + block_size - rest_columns;
            p_out = send_block + offset + block_size - rest_columns;
            auxSum = 0.0;
            for (x = -1; x <= 0; x++) {
                for (y = -1; y <= 1; y++) {
                    auxSum += kernel[x + 1][y + 1] * *(p_in + x * rest_columns + y);
                }
            }
            *p_out = (unsigned char) auxSum;
            
            MPI_Send(send_block, block_size, MPI_UNSIGNED_CHAR, MASTER_RANK, num_cycles, MPI_COMM_WORLD);
        }

        gettimeofday(&t2, NULL);
        printf("Process: %d; Convolutions: %lf\n", myrank, (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6));
        
        free(send_block);
        free(recv_block);
    }

    free(kernel);
    if (f_out != NULL) {
        free(f_out);
    }
    
    // Termination of MPI communication
    MPI_Finalize();

    exit(EXIT_SUCCESS);
}