python src/eval_et_bench.py \
    --ckpt_path /path/to/model \
    --output_path ./eval_et_bench.json

python data/eval/et_bench/compute_metrics.py \
    ./eval_et_bench.json
