source env/bin/activate

git clone https://github.com/baaivision/EVA.git

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

pip install ninja timm
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./

