# helper script used by spin-up-vm.sh and babysit-tpu-vm.sh


# Args we want
# VM name
# rest are flags:
# zone (default us-east1-d)
# type of box (default: v3-32
# vm image (default: tpu-vm-base)
# preemptible (default: false)
# autodelete (default: true)
# setup script (default: infra/setup-tpu-vm.sh)


# set defaults
ZONE="us-east1-d"
TYPE="v3-32"
VM_IMAGE="tpu-vm-base"
PREEMPTIBLE=false
AUTODELETE=true
SETUP_SCRIPT="$SCRIPT_DIR/setup-tpu-vm.sh"

# parse args
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -z|--zone)
      ZONE="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--type)
      TYPE="$2"
      shift # past argument
      shift # past value
      ;;
    -i|--image)
      VM_IMAGE="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--preemptible)
      PREEMPTIBLE="true"
      shift # past argument
      ;;
    -a|--autodelete)
      AUTODELETE="false"
      shift # past argument
      ;;
    -s|--setup)
      SETUP_SCRIPT="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option, assume it's the vm name
      # error out if we already set a name
      if [ -n "$VM_NAME" ]; then
        echo "Error: VM name already set to $VM_NAME"
        exit 1
      fi
      VM_NAME="$1"
      shift # past argument
      ;;
  esac
done
