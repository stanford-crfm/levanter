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


### Running a job on Slurm

```bash
srun --account=nlp --cpus-per-task=128 --gres=gpu:8 --job-name=levanter-multi-1 --mem=1000G --nodelist=sphinx[7-8] --open-mode=append --partition=sphinx --time=14-0 ~/src/levanter/scripts/launch.sh python examples/gpt2_example.py --config_path config/gpt2_small.yaml --data.cache_dir /scr-ssd/dlwh/tokenized/wikitext --trainer.per_device_parallelism 8 --model.resid_pdrop 0.0 --model.attn_pdrop 0.0 --model.seq_len 512
```