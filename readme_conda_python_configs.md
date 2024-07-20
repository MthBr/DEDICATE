dedicate_project
==============================

#This readme contains a sequence of commands to install all the necessary packages.
#Essentialy start with a conda environment, install all the packages, and the dedicate_code structure.
# in case you can/want to use Nvidia GPU install AmgX  and install pyamgx manually




#https://stackoverflow.com/questions/501940/simple-simulations-for-physics-in-python


conda update -n base -c defaults conda


conda clean --packages --tarballs
conda clean --all
conda update --all
conda update conda
conda update conda-build


conda list --revisions

conda activate base
conda install --revision 0

conda info
conda config --show-sources
conda list --show-channel-urls




# how to create developing environment [env]
conda env create -f DEDICATE-environment.yml 
#NOTE: it may take some time to solve all the dependencies!

# how to remove developing environment [env]
conda env remove -n  dedicate-env
conda env list

# how to install package, in case of failure of pip [env]
conda activate dedicate-env
cd DEDICATE
pip install -e .

conda deactivate


# AMGX#bindings
#https://github.com/NVIDIA/AMGX/releases
sudo apt  install cmake

# Please set the CUDAToolkit_ROOT
cd AMGX-2.4.0
mkdir build
cd build
cmake ../

cmake -DCMAKE_C_COMPILER=gcc -DMAKE_NO_MPI=True -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH="80" ..
#or 
cmake -DMAKE_NO_MPI=True -DCUDA_ARCH="90" ..

make -j16 all
#or
make -j16 install


#TODO
#nvcc warning : The 'compute_35', 'compute_37', 'compute_50', 'sm_35', 'sm_37' and 'sm_50' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).




# Building and installing pyamgx (download it pyamgx.readthedocs.io)
conda activate dedicate-env
export AMGX_DIR=/home/quasalab/Projects_LAB/AMGX-2.4.0
# export AMGX_BUILD_DIR=$AMGX_DIR/build
cd pyamgx-main
pip install .









# how to install  https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda-repo-ubuntu2204-12-5-local_12.5.1-555.42.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-5-local_12.5.1-555.42.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install cuda-toolkit
sudo apt-get install nvidia-gds
sudo reboot

export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}



Add these line to your ~/.bashrc and reload the terminal. After that cmake will find the nvcc
# cuda 10.2
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin




#https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#switching-between-driver-module-flavors

#To switch from legacy to open:  

sudo apt-get --purge remove nvidia-kernel-source-XXX
sudo apt-get install --verbose-versions nvidia-kernel-open-XXX
sudo apt-get install --verbose-versions cuda-drivers-XXX
sudo apt-get install -y nvidia-driver-555-open
sudo apt-get install -y cuda-drivers-555


# To switch from open to legacy:
sudo apt-get remove --purge nvidia-kernel-open-XXX
sudo apt-get install --verbose-versions cuda-drivers


















