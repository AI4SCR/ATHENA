# Install ATHENA

```{code-block}
# create a new virtual environment with Python 3.8
conda create -y -n athena python=3.8
conda activate athena

# install spatialOmics data container
pip install "git+git://github.com/histocartography/spatial-omics.git@master"

# install ATHENA package
pip install "git+git://github.com/histocartography/athena.git@master"
```