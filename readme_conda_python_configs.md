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


# AMGX installing..  #buildings
cd AMGX-2.2.0
mkdir build
cd build
cmake ../
make -j16 all


#TODO
#nvcc warning : The 'compute_35', 'compute_37', 'compute_50', 'sm_35', 'sm_37' and 'sm_50' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).




# Building and installing pyamgx (download it pyamgx.readthedocs.io)
conda activate dedicate-env
export AMGX_DIR=/home/modal/Projets_local/AMGX-2.2.0
# export AMGX_BUILD_DIR=$AMGX_DIR/build
cd pyamgx-main
pip install .

