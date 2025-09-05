#!/bin/bash
set -e  # Exit on error

echo "🚀 Starting CUDA 12.6 + PyTorch + Apex install on Ubuntu 24.04..."

### STEP 1: Remove old CUDA
echo "🔧 Removing old CUDA versions..."
sudo apt-get --purge remove -y $(dpkg -l | grep -E 'cuda|cublas' | awk '{print $2}') || true
sudo apt-get autoremove -y
sudo apt-get autoclean

### STEP 2: Add CUDA 12.6 repo (use 22.04 packages on 24.04)
echo "🔗 Adding CUDA 12.6 repo (targeting Ubuntu 22.04 packages)..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Use keyring instead of deprecated apt-key
sudo mkdir -p /usr/share/keyrings
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor | sudo tee /usr/share/keyrings/cuda-archive-keyring.gpg > /dev/null

# Add the repo
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2204.list

sudo apt-get update

### STEP 3: Install CUDA toolkit
echo "📦 Installing CUDA toolkit 12.6..."
sudo apt-get -y install cuda-toolkit-12-6

### STEP 4: Set up environment variables
echo "🧩 Configuring environment variables..."
if ! grep -q "cuda-12.6" ~/.bashrc; then
  echo "export PATH=/usr/local/cuda-12.6/bin:\$PATH" >> ~/.bashrc
  echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
fi

### STEP 5: Verify CUDA
echo "🔍 Verifying CUDA installation..."
nvcc --version || echo "⚠️  nvcc not found in PATH yet (restart or source ~/.bashrc)"

### STEP 6: Install NVIDIA driver
echo "🖥️  Installing NVIDIA driver..."
sudo ubuntu-drivers autoinstall

### STEP 7: Activate Python environment
echo "🐍 Activating Python virtual environment..."
source env/bin/activate

### STEP 8: Install PyTorch with CUDA 12.6 support
echo "🔥 Installing PyTorch with CUDA 12.6 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

### STEP 9: Install Apex
echo "⚙️  Installing Apex..."
if [ ! -d "apex" ]; then
  git clone https://github.com/NVIDIA/apex.git
fi

cd apex
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir .
python setup.py install --cuda_ext --cpp_ext
cd ..

echo ""
echo "✅ CUDA 12.6, PyTorch, and Apex installation complete!"
echo "➡️  Please run 'source ~/.bashrc' or restart your terminal to activate CUDA environment variables."
echo "🔁 If a new NVIDIA driver was installed, consider rebooting: sudo reboot"
