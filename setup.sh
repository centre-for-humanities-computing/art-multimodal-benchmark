# create virtual environment
python3 -m venv env

# activate env
source ./env/bin/activate

# install required packages
pip install -r requirements.txt

# explicitly install ipykernel as well
pip install ipykernel

# install kernel so env can be used in jupyter notebooks
python -m ipykernel install --user --name=benchmark_env

deactivate 