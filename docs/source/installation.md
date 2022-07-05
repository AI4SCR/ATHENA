# Install ATHENA

```{code-block}
# create a new virtual environment with Python 3.8
conda create -y -n athena python=3.8
conda activate athena

# install athena, spatial omics
pip install ai4scr-spatial-omics ai4scr-athena

# install interactive tools
pip jupyterlab
```