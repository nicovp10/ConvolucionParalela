#!/usr/bin/env python3


import matplotlib.pyplot as plt



def main():
    """Main function"""

    # VARIABLES
    n = range(1, 33)
    C_values = [2, 25, 100]
    ITER = 5


    # Read the files
    for i in n:
        filename = 'output_files/output_' + str(i) + '.txt'
        with open(filename, 'r') as datafile:

            # Check if the start of the file is correct
            line = datafile.readline()
            if line != 'C: ' + str(C_values[0]) + '\n':    
                print('ERROR: C value ' + str(C_values[0]) + ' not found at the start of the file ' + filename + '\n')
                exit()
            
            
            


# START OF EXECUTION
if __name__ == '__main__':
    main()