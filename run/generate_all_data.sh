set -e

start=0
end=4

for seed in $(seq $start $end)
do
    tag=s$seed
    echo ------------------------------------------
    echo On seed $seed w/ tag $tag
    python scripts/generate_data.py seed=$seed tag=$tag
done
