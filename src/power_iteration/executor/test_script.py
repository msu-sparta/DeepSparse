import os
import shutil

#edit the list below to include your own matrices
MATRICES=["inline_1","HV15R","Queen_4147","Nm7"]
#MATRICES=["twitter7","sk-2005"]
#BLOCK_SIZES=["1024","2048","4096","8192","16384","32768","65536"]
BLOCK_SIZES=["16384","32768","65536"]
#BLOCK_SIZES=["131072","262144","524288","1048576"]

#edit the variable below to point your own matrix folder
#matrix_folder = "/mnt/home/alperena/DeepSparse/deepsparse/Matrices/NERSC/"
matrix_folder = "/mnt/gs18/scratch/users/alperena/MATRICES/NERSC/"

for MATRIX in MATRICES:

    for BLOCK_SIZE in BLOCK_SIZES:

        #print("MATRIX: " + MATRIX + "\tblock count: " + BLOCK_SIZE)

        print("\nexecuting libcsr")
        libcsr_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/power_libcsr_executor.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10"
        os.system(libcsr_exe);

        print("\nexecuting libcsb")
        libcsb_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/power_libcsb_executor.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10"
        #libcsb_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/power_libcsb_executor_intCSB.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10"
        os.system(libcsb_exe);

        print("\nexecuting global")
        dag_gen = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/power_global.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " " + MATRIX + "_" + BLOCK_SIZE
        dag_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/power_global_executor.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10 " + MATRIX + "_" + BLOCK_SIZE + "*"
        #dag_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/power_global_executor_intCSB.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10 " + MATRIX + "_" + BLOCK_SIZE + "*"
        os.system(dag_gen)
        os.system(dag_exe)

        print("\nexecuting hpx")
        hpx_exe = "./../exe/my_hpx_program --iterations 10 --block_width " + BLOCK_SIZE + " --matrix_file " + matrix_folder + MATRIX + ".cus --hpx:threads 14"
        hpx_exe = "./../exe/my_hpx_program_intCSB --iterations 10 --block_width " + BLOCK_SIZE + " --matrix_file " + matrix_folder + MATRIX + ".cus --hpx:threads 14"
        os.system(hpx_exe)

        #print("\n")

MATRIX=["inline_1", "HV15R", "Queen_4147", "Nm7", "it-2004", "twitter7", "sk-2005"]
MATRIX_INFO=["-n 503712 -nnz 18660027 -graph ~/DeepSparse/deepsparse/Matrices/NERSC/inline_1.cus",
        "-n 2017169 -nnz 283073458 -graph ~/DeepSparse/deepsparse/Matrices/NERSC/HV15R.cus",
        "-n 4147110 -nnz 166823197 -graph ~/DeepSparse/deepsparse/Matrices/NERSC/Queen_4147.cus",
        "-n 4985422 -nnz 647663919 -graph ~/DeepSparse/deepsparse/Matrices/NERSC/Nm7.cus",
        "-n 41291594 -nnz 1150725436 -graph /mnt/gs18/scratch/users/alperena/MATRICES/NERSC/it-2004.cus",
        "-n 41652230 -nnz 1468365182 -graph /mnt/gs18/scratch/users/alperena/MATRICES/NERSC/twitter7.cus",
        "-n 50636154 -nnz 1949412601 -graph /mnt/gs18/scratch/users/alperena/MATRICES/NERSC/sk-2005.cus"]
BLOCK_SIZES=["8192","16384","32768","65536"]
#BLOCK_SIZES=["262144","524288","1048576"]
#BLOCK_SIZES=["2097152", "4194304"]

for i in range(4, 7):
    print("\nmatrix name: " + MATRIX[i])
    for BLOCK_SIZE in BLOCK_SIZES:
        print("\nmatrix size: " + MATRIX[i] + "\tblock size: " + BLOCK_SIZE)
        sparse_regent =  "~/legion/language/regent.py ../exe/new_regent_intCSB.rg " + MATRIX_INFO[i] +  " -b " + BLOCK_SIZE + " -i 10 -ll:cpu 12 -ll:util 2 -ll:csize 80200"
        os.system(sparse_regent)
