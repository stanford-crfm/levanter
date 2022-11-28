```bash
# 3.11 is too new for tensorstore
conda create --name levanter -c nvidia -c conda-forge jax  cuda-toolkit python=3.10
conda activate levanter
pip install -r requirements.txt
```

```bash
# optional: install go to get gpu heap utilization
sudo apt-get install golang
# or do this if you don't have sudo: 
conda install -c conda-forge go
```

