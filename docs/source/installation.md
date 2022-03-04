# Install ATHENA

```{code-block}
# create a new virtual environment with Python 3.8
conda create -y -n athena python=3.8
conda activate athena

# install spatialOmics data container
pip install "git+https://github.com/AI4SCR/spatial-omics.git@master"

# install ATHENA package
pip install "git+https://github.com/AI4SCR/ATHENA.git@master"
```