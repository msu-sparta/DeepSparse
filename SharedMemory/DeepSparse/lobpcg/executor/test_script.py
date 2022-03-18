import os
import shutil

MATRICES=["inline_1","HV15R","Queen_4147","Nm7"]
#MATRICES=["it-2004","twitter7","sk-2005"]
#BLOCK_SIZES=["1024","2048","4096","8192","16384","32768","65536"]
BLOCK_SIZES=["65536"]

#edit the variable below to point your own matrix folder
matrix_folder = "/mnt/home/alperena/DeepSparse/deepsparse/Matrices/NERSC/"
#matrix_folder = "/mnt/gs18/scratch/users/alperena/MATRICES/NERSC/"

for MATRIX in MATRICES:
    for BLOCK_SIZE in BLOCK_SIZES:
        print("MATRIX: " + MATRIX + "\tblock count: " + BLOCK_SIZE)

        print("\nexecuting libcsr")
        libcsr_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/lobpcg_libcsr_executor.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10 "
        #+ "> experiment_lobpcg_libcsr_" + MATRIX + "_" + BLOCK_SIZE + ".out"
        os.system(libcsr_exe);

        print("\nexecuting libcsb")
        libcsb_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/lobpcg_libcsb_executor.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10 "
        #+ "> experiment_lobpcg_libcsb_" + MATRIX + "_" + BLOCK_SIZE + ".out"
        os.system(libcsb_exe);

        print("\nexecuting global")
        global_gen = "./../exe/lobpcg_global.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10 " + MATRIX + "_" + BLOCK_SIZE
        nonloop = MATRIX + "_" + BLOCK_SIZE + "*nonloop*txt "
        firstloop = MATRIX + "_" + BLOCK_SIZE + "*firstloop*txt "
        secondloop = MATRIX + "_" + BLOCK_SIZE + "*secondloop*txt "
        global_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/lobpcg_global_executor.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10 " + nonloop + firstloop + secondloop
        #+ "> experiment_lobpcg_global_" + MATRIX + "_" + BLOCK_SIZE + ".out"
        os.system(global_gen)
        os.system(global_exe)

        print("\n")
