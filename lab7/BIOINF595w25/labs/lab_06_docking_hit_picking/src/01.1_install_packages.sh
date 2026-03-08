


pip install torch torchvision torchaudio 

pip install rdkit

pip install unimol_tools
# Note: for me it tried to recompile and downgrade pandas
# and then installing the next ones upgrade pandas again, so far this doesn't seem to be an issue
# but if needed, these could be installed into different conda environments

# may need to be in it's own conda env
pip install umap-learn
pip install datashader
pip install holoviews
pip install bokeh
pip install jupyterlab
pip install jupyter_bokeh
pip install pyarrow

# packages for clustering
pip install scanpy
pip install leidenalg


# install the ggrastr package
Rscript -e 'install.packages("ggrastr", repos="https://cloud.r-project.org")'


###########

#To get the molecule embeddings to work locally
conda create --name=main39 --yes
conda activate main39
conda install --name=main39 python=3.9 --yes

pip install datashader
pip install holoviews
pip install bokeh
pip install jupyterlab
pip install jupyter_bokeh
pip install pyarrow

pip install ipykernel
ipython kernel install --user --name=main39

# copy


jupyter lab
# in jupter lab
