# create virtual environment
/opt/homebrew/bin/python3.11 -m venv env

#python3.11 -m venv env

# activate env
source env/bin/activate

#CPU
pip install tensorflow==2.19.0 

# install required packages
pip install -r requirements.txt

# GPU
#pip install tensorflow[and-cuda]

# explicitly install ipykernel as well
pip install ipykernel

# install kernel so env can be used in jupyter notebooks
python -m ipykernel install --user --name=benchmark_env # need to restart vscode to use!

deactivate 