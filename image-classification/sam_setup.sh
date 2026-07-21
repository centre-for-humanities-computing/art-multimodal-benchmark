sudo apt-get update
sudo apt-get install python3-venv
python3 -m venv .venv
source .venv/bin/activate

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

rm -rf sam3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install -e ".[notebooks]"
pip install -e ".[train,dev]"

pip install einops ninja
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git --no-build-isolation