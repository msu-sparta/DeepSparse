import os
import shutil

#edit the list below to include your own matrices
MATRICES=["inline_1","HV15R","Queen_4147","Nm7"]
#MATRICES=["it-2004","twitter7","sk-2005"]
BLOCK_SIZES=["1024","2048","4096","8192","16384","32768","65536"]

#edit the variable below to point your own matrix folder
matrix_folder = "/mnt/home/alperena/DeepSparse/deepsparse/Matrices/NERSC/"
#matrix_folder = "/mnt/gs18/scratch/users/alperena/MATRICES/NERSC/"

for BLOCK_SIZE in BLOCK_SIZES:
    for MATRIX in MATRICES:
        #print("MATRIX: " + MATRIX + "\tblock count: " + BLOCK_SIZE)

        print("\nexecuting libcsr")
        libcsr_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/lanczos_libcsr_executor.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10 " + "> experiment_lanczos_libcsr_" + MATRIX + "_" + BLOCK_SIZE + ".out"
        os.system(libcsr_exe);

        print("\nexecuting libcsb")
        libcsb_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/lanczos_libcsb_executor.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10 " + "> experiment_lanczos_libcsb_" + MATRIX + "_" + BLOCK_SIZE + ".out"
        os.system(libcsb_exe);

        print("\nexecuting global")
        global_gen = "./../exe/lanczos_global.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10 " + MATRIX + "_" + BLOCK_SIZE
        global_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/lanczos_global_executor.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10 " + MATRIX + "_" + BLOCK_SIZE + "* " + "> experiment_lanczos_global_" + MATRIX + "_" + BLOCK_SIZE + ".out"
        os.system(global_gen)
        os.system(global_exe)
