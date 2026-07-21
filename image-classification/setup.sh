sudo apt-get update

sudo apt-get install python3-venv

# Create virtual environment in ./env
python3 -m venv env

# Activate the environment
source env/bin/activate

python -V

pip install --upgrade pip

# install required packages
pip install -r requirements.txt

# explicitly install ipykernel as well
pip install ipykernel

# install kernel so env can be used in jupyter notebooks
python -m ipykernel install --user --name=benchmark_env # need to restart vscode to use!

deactivate 