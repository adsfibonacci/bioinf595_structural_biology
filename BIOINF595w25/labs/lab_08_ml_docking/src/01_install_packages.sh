# The installation of NeuralPlexer is a little bit tricky
# The challenge is to make sure the versions of all the relevant packages
# are working well together. Here is the setup I was able to get to work
#    GPU: Tesla V100-PCIE-16GB
#    CUDA Version: 12.6
#    CUDNN Version: 12.6-v9.6.0
#    pytorch: 2.1.2
#    openfold: 2.0.0  (pl_upgrades branch circa 3/20/2025)
#    neuralplexer: 0.1.0

# NeuralPlexer has openfold as a dependency, but it doesn't seem to install
# cleanly. What I had to do was first install a specific version of flash-attn,
# then openfold, then NeuralPlexer. So try doing these steps by creating an environment
# and installing these packages in order and reach out if you get stuck.


echo "Please don't run this as a script, instead run it line-by-line"
exit 1

############################################################
# Set up modules and turbo for installing packages locally #
############################################################

module purge
module load cuda/12.6.3
module load cudnn/12.6-v9.6.0
module save bioinf595
# now to get this named collection modules in a shell:
#
#  module restore bioinf595
#


# Create a symlink for the class turbo into the home directory
ln -s /nfs/turbo/dcmb-class/bioinf595/sec001/${USER} ${HOME}/turbo_bioinf595
# this allows you to e.g. `cd ~/turbo_bioinf595` and the data will live
# on turbo not in your home directory. you can check symlinks with
# `ls -ls ${HOME}/turbo/bioinf595`


# create an opt directory where you can install packages
mkdir ${HOME}/turbo_bioinf595/opt




###################################
# Clean out all versions of conda #
###################################
# Note that if you have conda setup for other research this will remove those environments!
# If you haven't installed mamba or conda environments you won't need to do this, but you can
# check by looking in your ${HOME}/.bashrc for these blocks.

# Remove conda initialize block from ${HOME}/.bashrc
sed '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' ${HOME}/.bashrc > ${HOME}/.bashrc_new
# check that ${HOME}/.bashrc_new looks ok
mv ${HOME}/.bashrc_new ${HOME}/.bashrc

# Remove mamba initialize block from ${HOME}/.bashrc
sed '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' ${HOME}/.bashrc > ${HOME}/.bashrc_new
# check that ${HOME}/.bashrc_new looks ok
mv ${HOME}/.bashrc_new ${HOME}/.bashrc

rm -f ${HOME}/.condarc
rm -rf .conda


#########################################
# Install a local version of miniforge3 #
#########################################
# https://github.com/conda-forge/miniforge
# Miniforge is a version of conda and mamba produced by conda-forge (an open source community)
# not Anaconda (a for-profit company)

# I recommend moving conda out of your home directory as there is often not enough space to
# have it in home directory.
cd ${HOME}/turbo_bioinf595/opt
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b -u -p ${HOME}/turbo_bioinf595/opt/miniforge3
#  '-b' => batch mode, accept terms of service
#  '-u' => update an existing installation
#  '-p PREFIX' => installation prefix
rm -rf Miniforge3-latest-Linux-x86_64.sh

# activate the local conda environment
source ${HOME}/turbo_bioinf595/opt/miniforge3/etc/profile.d/conda.sh
source ${HOME}/turbo_bioinf595/opt/miniforge3/etc/profile.d/mamba.sh
conda activate

conda init
mamba init
# restart shell

conda update -n base --all --yes


################################
# Create bioinf595 environment #
################################
mamba create -n bioinf595

# check that the environment was created:
mamba env list


#
conda activate bioinf595


####################
# install openfold #
####################
cd ${HOME}/turbo_bioinf595/opt
# https://openfold.readthedocs.io/en/latest/Installation.html
# note we're using the pl_upgrades branch because we have cuda version 12
git clone -b pl_upgrades https://github.com/aqlaboratory/openfold.git
cd ${HOME}/turbo_bioinf595/opt/openfold

# based on this issue: https://github.com/aqlaboratory/openfold/issues/519
# edit environment.yml to specify specific version of flash-attn
sed "s/      - flash-attn/      - flash-attn==2.6.3/g" environment.yml > environment_patched.yml

# install openfold
mamba env update -n bioinf595 -f environment_patched.yml --yes
pip install -e .



# now instlal NeuralPLexer
cd ${HOME}/turbo_bioinf595/opt
git clone https://github.com/zrqiao/NeuralPLexer.git
cd ${HOME}/turbo_bioinf595/opt/NeuralPLexer

# remove openfold because we've already installed a specific version of it
cat requirements.txt | sed '/openfold/d' > requirements_new.txt
mv requirements_new.txt > requirements.txt
mamba env update -n bioinf595 -f environment_dev.yaml --yes
pip install -e .


# download model weights and test data from zenodo
wget https://zenodo.org/records/10373581/files/neuralplexermodels_downstream_datasets_predictions.zip
unzip neuralplexermodels_downstream_datasets_predictions.zip

# keep only the 'complex_structure_prediction.ckpt' model weights
rm -rf neuralplexermodels_downstream_datasets_predictions.zip
rm -rf neuralplexermodels_downstream_datasets_predictions/downstream_test_datasets
rm -rf neuralplexermodels_downstream_datasets_predictions/predictions
rm -rf neuralplexermodels_downstream_datasets_predictions/models/pdbbind_finetuned
cd ../..


##############################
# Download and install smina #
##############################

mkdir -p ${HOME}/turbo_bioinf595/opt/bin
cd ${HOME}/turbo_bioinf595/opt/bin
wget https://sourceforge.net/projects/smina/files/smina.static/download -O smina && chmod +x smina
cd ..
