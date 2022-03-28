# DeepSparseGPU
##### Programming Model: OpenMP target offload with custom memory manager

## Module Required
Load the following modules on a compute node on CoriGPU:

module load cdt/19.11                                                       
module swap PrgEnv-intel PrgEnv-cray                                        
module switch cce cce/9.1.0-classic                                         
module swap craype-haswell craype-x86-skylake                               
module unload cray-libsci                                                   
module load cuda
module load cudatoolkit craype-accel-nvidia70     
module unload darshan/3.2.1
module unload cray-mpich/7.7.10

## Building the project
Go to 'DeepSparseGPU/OpenMP_MemoryManager/scripts' folder and run the build-on-corigpu.sh script to compile the project on CoriGPU.
