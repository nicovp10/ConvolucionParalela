#!/usr/bin/env python3


import matplotlib.pyplot as plt



def main():
    """Main function"""

    # VARIABLES
    num_proc = range(1, 33)
    C_values = [2, 25, 100]
    ITER = 5

    final_times = [[0 for _ in range(len(num_proc))] for _ in range(len(C_values))]


    # Read the files
    for n in num_proc:
        filename = 'output_files/output_' + str(n) + '.txt'
        with open(filename, 'r') as datafile:
            for i in range(len(C_values)):

                # Initialize the lists
                sends_times = []
                bcasts_times = []
                recvs_times = []
                process_convolutions_times = []
                for it in range(ITER):
                    process_convolutions_times.append([0]*n)
                
                # Skip lines until the corresponding value of C is found
                line = datafile.readline()
                while line:
                    if line == 'C: ' + str(C_values[i]) + '\n':    
                        break
                    line = datafile.readline()
                    
                # Skip lines until the first iteration is found
                line = datafile.readline()
                while line:
                    if line == 'iteration: 1\n':    
                        break
                    line = datafile.readline()


                # Read the data of the ITER iterations for more than one process
                for it in range(ITER):

                    # Skip lines until the init of the information
                    line = datafile.readline()
                    while line:
                        if line == 'Init\n':    
                            break
                        line = datafile.readline()
                    

                    # Extract the data
                    if n > 1:
                        # Sends: X
                        line = datafile.readline()
                        sends_times.append(float(line.split(':')[1].strip()))

                        # Bcast: X
                        line = datafile.readline()
                        bcasts_times.append(float(line.split(':')[1].strip()))

                    # Process X; Convolutions X
                    for j in range(n):
                        line = datafile.readline()
                        info = line.split(';')
                        proc = int(info[0].split(':')[1].strip())
                        process_convolutions_times[it][j] = float(info[1].split(':')[1].strip())
                    
                    if n > 1:
                        # Recvs: X
                        line = datafile.readline()
                        recvs_times.append(float(line.split(':')[1].strip()))

                    # Read the line "iteration: X"
                    if it < ITER - 1:
                        datafile.readline()


                # Calculate the best time value and add it to the final values
                max_conv_times = []
                for j in range(ITER):
                    max_conv_times.append(max(process_convolutions_times[j]))
                best_time = min(max_conv_times)
                best_it = max_conv_times.index(best_time)
                if n > 1:
                    best_time += sends_times[best_it] + bcasts_times[best_it] + recvs_times[best_it]
                
                final_times[i][n-1] = best_time
    

    # Plots
    print(final_times)




# START OF EXECUTION
if __name__ == '__main__':
    main()