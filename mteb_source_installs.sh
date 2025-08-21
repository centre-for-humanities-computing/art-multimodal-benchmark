source ./env/bin/activate

pip install -r mteb_reqs.txt

# Clone EVA repo for QuanSun/EVA02 models
git clone git@github.com:baaivision/EVA.git

# Install ninja and timm via pip
pip install ninja timm

# Install xformers from source (required for EVA) - Only works on machines with NVIDIA and GPU compatability
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

# Clone and install NVIDIA apex for mixed precision (required for EVA)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
cd ..