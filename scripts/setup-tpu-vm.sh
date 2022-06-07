set -e
# broadly based on https://github.com/ayaka14732/tpu-starter

sudo apt update
sudo apt upgrade -y

# python 310
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.10-full python3.10-dev

python3.10 -m venv $HOME/.venv310
source $HOME/.venv310/bin/activate

pip install -U pip
pip install -U wheel

# jax
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# TF
#wget https://gist.github.com/ayaka14732/4954f64b7246beafabb45b636d96e92a/raw/d518753d166f3b77009d1f228101d93ff733d0d2/tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl.0 https://gist.github.com/ayaka14732/4954f64b7246beafabb45b636d96e92a/raw/d518753d166f3b77009d1f228101d93ff733d0d2/tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl.1
#cat tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl.0 tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl.1 > tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl
#rm -f tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl.0 tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl.1
#pip install tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl


git clone https://github.com/stanford-crfm/psithuros
cd psithuros
