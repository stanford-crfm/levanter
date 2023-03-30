set -x
# broadly based on https://github.com/ayaka14732/tpu-starter

# tcmalloc interferes with intellij remote ide
sudo patch -f -b /etc/environment << EOF
2c2
< LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
---
> #LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
EOF

# don't complain if already applied
retCode=$?
[[ $retCode -le 1 ]] || exit $retCode

# install python 3.10, latest git, and nfs
#sudo apt-get install -y software-properties-common
#sudo add-apt-repository -y ppa:deadsnakes/ppa
#sudo add-apt-repository -y ppa:git-core/ppa
#sudo apt-get update
#sudo apt-get install -y python3.10-full python3.10-dev nfs-common git golang

sudo systemctl stop unattended-upgrades  # this frequently holds the apt lock
sudo systemctl disable unattended-upgrades
sudo apt remove -y unattended-upgrades
# if it's still running somehow, kill it
if [ $(ps aux | grep unattended-upgrade | wc -l) -gt 1 ]; then
  sudo kill -9 $(ps aux | grep unattended-upgrade | awk '{print $2}')
fi
# sometimes apt-get update fails, so retry a few times
for i in {1..5}; do
  sudo apt-get install -y software-properties-common \
  && sudo add-apt-repository -y ppa:deadsnakes/ppa \
  && sudo add-apt-repository -y ppa:git-core/ppa \
  && sudo apt-get update \
  && sudo apt-get install -y python3.10-full python3.10-dev nfs-common git \
  && break
done
sudo systemctl start unattended-upgrades

# set up nfs
NFS_SERVER=10.5.220.250
MOUNT_POINT="/files"
sudo mkdir -p ${MOUNT_POINT}
CURRENT_NFS_ENTRY=$(grep ${NFS_SERVER} /etc/fstab)
DESIRED_NFS_ENTRY="${NFS_SERVER}:/propulsion ${MOUNT_POINT} nfs defaults 0 0"
# if different, fix
if [ "$CURRENT_NFS_ENTRY" != "$DESIRED_NFS_ENTRY" ]; then
  set -e
  echo "Setting up nfs"
  grep -v "${NFS_SERVER}" /etc/fstab > /tmp/fstab.new
  echo "${DESIRED_NFS_ENTRY}" >> /tmp/fstab.new
  # then move the new fstab back into place
  sudo cp /etc/fstab /etc/fstab.orig
  sudo mv /tmp/fstab.new /etc/fstab
fi
sudo mount -a


# default to loading the venv
sudo bash -c "echo \"source ${MOUNT_POINT}/venv310/bin/activate\" > /etc/profile.d/activate_shared_venv.sh"

for x in `ls -d /files/lev*`; do
  git config --global --add safe.directory $x
done

# symlink lev* to home
ln -s /files/lev* ~
