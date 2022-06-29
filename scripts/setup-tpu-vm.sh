# broadly based on https://github.com/ayaka14732/tpu-starter

# tcmalloc interferes with intellij remote ide
sudo yes no | patch -b /etc/environment << EOF
2c2
< LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
---
> #LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
EOF

# don't complain if already applied
retCode=$?
[[ $retCode -le 1 ]] || exit $retCode

#sudo apt update
#sudo apt upgrade -y
#
## python 310
#sudo apt install -y software-properties-common
#sudo add-apt-repository -y ppa:deadsnakes/ppa
#sudo apt install -y python3.10-full python3.10-dev nfs-common

sudo bash -c 'echo "source /files/venv310/bin/activate" >> /etc/profile.d/activate_shared_venv.sh'
