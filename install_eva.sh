# Clone EVA repo:
git clone https://github.com/baaivision/EVA.git

source env/bin/activate
# Install xformers from source (required for EVA) - Only works on machines with NVIDIA and GPU compatability
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

# we need to access EVA-CLIP loaders in the subdirectory "EVA-CLIP", but the dash causes problems with sys paths;
# renaming the EVA-CLIP folder to EVA_CLIP by creating a symlink between them to ensure compatability with MTEB

# Rename EVA-CLIP folder to EVA_CLIP
mv EVA/EVA-CLIP EVA/EVA_CLIP

# Create a symlink named EVA-CLIP pointing to EVA_CLIP
ln -s $(pwd)/EVA/EVA_CLIP $(pwd)/EVA/EVA-CLIP

# do same thing for all EVA folders

mv EVA/EVA-01 EVA/EVA_01

ln -s $(pwd)/EVA/EVA_01 $(pwd)/EVA/EVA-01

mv EVA/EVA-02 EVA/EVA_02

ln -s $(pwd)/EVA/EVA_02 $(pwd)/EVA/EVA-02

mv EVA/EVA-CLIP-18B EVA/EVA_CLIP_18B

ln -s $(pwd)/EVA/EVA_CLIP_18B $(pwd)/EVA/EVA-CLIP-18B

deactivate