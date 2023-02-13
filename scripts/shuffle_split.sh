# Usage: shuffle_split.sh -n <num_shards> [--prefix <prefix> --suffix <suffix> -c <compression> -d <decompression> -x <seed> -m <memGB>] <files>
# Description: Shuffles files and splits them into <num_shards> shards. Lines will not be split.
# Automatically applies decompression as necessary

# process args
while [ $# -gt 0 ]; do
    case "$1" in
        -n)
            num_shards=$2
            shift
            ;;
        --prefix)
            prefix=$2
            shift
            ;;
        --suffix)
            suffix=$2
            shift
            ;;
        -c)
            compression=$2
            shift
            ;;
        -d)
            decompression=$2
            shift
            ;;
        -x)
            SEED=$2
            shift
            ;;
        -m)
            mem=$2
            shift
            ;;
        -*)
            echo "Error: Unknown flag $1"
            exit 1
            ;;
        *)
            files="$files $1"
            ;;
    esac
    shift
done

# check args
if [ -z "$num_shards" ]; then
    echo "Error: num_shards not specified"
    exit 1
fi

if [ -z "$prefix" ]; then
  prefix="shard"
fi

if [ -z "$compression" ]; then
  if [ -z "$decompression" ]; then
    compression="none"
  else
    compression=$decompression
  fi
fi

if [ -z "$decompression" ]; then
    decompression=$compression
fi

if [ -z "$SEED" ]; then
    SEED=0
fi

if [ -z "$mem" ]; then
    mem=4
fi

# figure out command to cat each file and then to write compressed files (using split filter)
# we support gz, xz, zst, bz2, and uncompressed

if [ "$compression" = "gz" ]; then
    write_cmd="gzip -c"
elif [ "$compression" = "xz" ]; then
    write_cmd="xz -c"
elif [ "$compression" = "zst" ]; then
    write_cmd="zstd -c"
elif [ "$compression" = "bz2" ]; then
    write_cmd="bzip2 -c"
elif [ "$compression" = "none" ]; then
    write_cmd="cat"
else
    echo "Error: compression $compression not supported"
    exit 1
fi

if [ "$decompression" = "gz" ]; then
    cat_cmd="zcat"
elif [ "$decompression" = "xz" ]; then
    cat_cmd="xzcat"
elif [ "$decompression" = "zst" ]; then
    cat_cmd="zstdcat"
elif [ "$decompression" = "bz2" ]; then
    cat_cmd="bzcat"
elif [ "$decompression" = "none" ]; then
    cat_cmd="cat"
else
    echo "Error: decompression $decompression not supported"
    exit 1
fi

TEMP_FILE=$(mktemp)

# shuffle files
$cat_cmd $files | SEED=$SEED MEMORY=$mem terashuf > $TEMP_FILE

# split files
# determine how many lines to split

if [ "$compression" = "none" ]; then
    write_cmd_shuf="$write_cmd > \${FILE}"
else
    write_cmd_shuf="$write_cmd > \${FILE}.$compression"
fi

if [ -n "$suffix" ]; then
  extra_split_args="--additional-suffix=$suffix"
else
  extra_split_args=""
fi

split -n l/$num_shards -d -a3 --filter="$write_cmd_shuf" $extra_split_args $TEMP_FILE $prefix

# clean up
rm $TEMP_FILE
