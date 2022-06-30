VENV="/files/venv310"
# if the venv doesn't exist, make it
if [ ! -d "$VENV" ]; then
    echo "Creating virtualenv at $VENV"
    python3.10 -m venv $VENV
fi
source $VENV/bin/activate

pip install -U pip
pip install -U wheel

# jax
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html




# TF
#wget https://gist.github.com/ayaka14732/4954f64b7246beafabb45b636d96e92a/raw/d518753d166f3b77009d1f228101d93ff733d0d2/tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl.0 https://gist.github.com/ayaka14732/4954f64b7246beafabb45b636d96e92a/raw/d518753d166f3b77009d1f228101d93ff733d0d2/tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl.1
#cat tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl.0 tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl.1 > tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl
#rm -f tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl.0 tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl.1
#pip install tensorflow-2.10.0-cp310-cp310-linux_x86_64.whl
