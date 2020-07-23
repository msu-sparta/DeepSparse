import os
import shutil

#edit the list below to include your own matrices
MATRICES=["inline_1","HV15R","Queen_4147","Nm7"]
BLOCK_SIZES=["1024","2048","4096","8192","16384","32768","65536"]

#edit the variable below to point your own matrix folder
matrix_folder = "/mnt/home/alperena/DeepSparse/deepsparse/Matrices/NERSC/"

for MATRIX in MATRICES:
    
    for BLOCK_SIZE in BLOCK_SIZES:
        
        print("MATRIX: " + MATRIX + "\tblock count: " + BLOCK_SIZE)

        print("\nexecuting libcsr")
        libcsr_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/power_libcsr_executor.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10"
        os.system(libcsr_exe);

        print("\nexecuting libcsb")
        libcsb_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/power_libcsb_executor.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10"
        os.system(libcsb_exe);

        print("\nexecuting global")
        dag_gen = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/power_global.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " " + MATRIX + "_" + BLOCK_SIZE
        dag_exe = "OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close ./../exe/power_global_executor.x " + matrix_folder + MATRIX + ".cus " + BLOCK_SIZE + " 10 " + MATRIX + "_" + BLOCK_SIZE + "*"
        os.system(dag_gen)
        os.system(dag_exe)

        print("\n")
