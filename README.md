# mensa_t2i
SoTA T2I Adapting and Finetuning
--------------------------------------------------------------------------------------------------------------------
To start training run these commands:

chmod +x train_lora_job.sh
sbatch train_lora_job.sh


squeue --me
sacct -j 19735409 --format=JobID,State,Elapsed,MaxRSS
--------------------------------------------------------------------------------------------------------------------
To predict with a certain prompt cd into lora folder and run this:
python predict_lora.py \
  --prompt "Mensafood Chickpea polenta with ratatouille; sheep’s cheese with mint" \
  --num_images 2 \
  --height 512 \
  --width 512 \
  --lora_weights_dir ./lora-adapters-forth-train/checkpoint-15000
--------------------------------------------------------------------------------------------------------------------
python augment_dataset.py \
    --input_csv  ../data/meals_unique_mensafood.csv \
    --images_root ../data \
    --output_dir ../data/aug \
    --output_csv  meals_augmented.csv \
    --n_augs 4 -j 8
