#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
#sudo dpkg -i cuda-keyring_1.1-1_all.deb
#sudo apt-get update
#sudo apt-get -y install cuda-toolkit-12-8

###### RUN THESE SEQUENTIALLY IN THE SAME BASH SCRIPT
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export CUDACXX=$CUDA_HOME/bin/nvcc 

## VERIFY

which nvcc