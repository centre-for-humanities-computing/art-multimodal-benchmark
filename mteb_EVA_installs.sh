source ./env/bin/activate

# Clone EVA repo for QuanSun/EVA02 models with HTTPS (creates folder in main dir called EVA)
git clone https://github.com/baaivision/EVA.git



# Install ninja and timm via pip (move to requirements file instead?)
pip install ninja timm

# Install xformers from source (required for EVA) - Only works on machines with NVIDIA and GPU compatability
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

# Clone and install NVIDIA apex for mixed precision (required for EVA)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
cd ..

# we need to access EVA-CLIP loaders in the subdirectory "EVA-CLIP", but the dash causes problems with sys paths;
# renaming the EVA-CLIP folder to EVA_CLIP while creating a symlink between them to ensure compatability with MTEB

# Rename the folder for Python compatibility (if not done already)
mv /content/EVA/EVA-CLIP /content/EVA/EVA_CLIP

# Create a symlink named EVA-CLIP pointing to EVA_CLIP
ln -s /content/EVA/EVA_CLIP /content/EVA/EVA-CLIP