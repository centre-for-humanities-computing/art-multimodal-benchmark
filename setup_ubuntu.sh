# install python3.11
sudo apt-get update
sudo apt-get install software-properties-common -y

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt-get install python3.11 python3.11-venv python3.11-dev -y

# Create virtual environment
python3.11 -m venv env

source env/bin/activate

# Install packages
pip install -r requirements.txt
pip install ipykernel

# install kernel so env can be used in jupyter notebooks
python -m ipykernel install --user --name=benchmark_env # need to restart vscode to use!

# Deactivate (optional)
deactivate