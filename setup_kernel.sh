source env/bin/activate

# Install packages
pip install ipykernel

# install kernel so env can be used in jupyter notebooks
python -m ipykernel install --user --name=benchmark_env # need to restart vscode to use!

# Deactivate (optional)
deactivate