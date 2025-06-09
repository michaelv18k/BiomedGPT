#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8082

user_dir=/content/BiomedGPT/module
bpe_dir=/content/BiomedGPT/utils/BPE

# val or test
split=$1

data_dir=/content/BiomedGPT/datasets/finetuning/vqa-rad
data=${data_dir}/test.tsv
# ans2label_file=${data_dir}/trainval_ans2label_pubmedclip.pkl
ans2label_file=${data_dir}/trainval_ans2label.pkl

declare -a Scale=('base')

for scale in ${Scale[@]}; do
    if [[ $scale =~ "tiny" ]]; then
        patch_image_size=256
    elif [[ $scale =~ "medium" ]]; then
        patch_image_size=256
    elif [[ $scale =~ "base" ]]; then  
        patch_image_size=384
    fi

    # path=/content/drive/MyDrive/biomedgpt_VQA/checkpoints/tuned_checkpoints/VQA-RAD/base/55_0.04_5e-05_384/checkpoint_best.pt
    path=/content/drive/MyDrive/checkpoints/tuned_checkpoints/VQA-RAD/base/55_0.04_5e-05_384/checkpoint_best.pt
    # path=/content/drive/MyDrive/checkpoint_best.pt
    result_path=/content/drive/MyDrive/results/vqa_rad_beam/${scale}
    mkdir -p $result_path
    selected_cols=0,5,2,3,4

    log_file=${result_path}/${scale}".log"
    # log_file=${result_path}/"val_"${scale}".log"

    python3.9 /content/BiomedGPT/evaluate.py \
        ${data} \
        --path=${path} \
        --user-dir=${user_dir} \
        --task=vqa_gen \
        --batch-size=64 \
        --log-format=simple --log-interval=100 \
        --seed=7 \
        --gen-subset=${split} \
        --results-path=${result_path} \
        --fp16 \
        --ema-eval \
        --beam-search-vqa-eval \
        --beam=1 \
        --unnormalized \
        --temperature=1.0 \
        --num-workers=0 \
        --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\"}"
done