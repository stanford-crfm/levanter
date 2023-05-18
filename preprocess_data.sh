COMMAND="python scripts/cache_dataset.py --config_path config/data/redpajama_1t_source.yaml"

n=0
until [ "$n" -ge 5 ]
do
   $COMMAND
   n=$((n+1)) 
   sleep 15
done
