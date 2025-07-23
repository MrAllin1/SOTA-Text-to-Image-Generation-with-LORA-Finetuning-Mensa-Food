# mensa_t2i
SoTA T2I Adapting and Finetuning
--------------------------------------------------------------------------------------------------------------------

## train
To start training run these commands:

chmod +x train_lora_job.sh
sbatch train_lora_job.sh


squeue --me
sacct -j 19735468 --format=JobID,State,Elapsed,MaxRSS
--------------------------------------------------------------------------------------------------------------------
To predict with a certain prompt cd into lora folder and run this:
python predict_lora.py \
  --prompt "Mensafood Chickpea polenta with ratatouille; sheep’s cheese with mint" \
  --num_images 2 \
  --height 512 \
  --width 512 \
  --lora_weights_dir ./lora-adapters-forth-train/checkpoint-15000
--------------------------------------------------------------------------------------------------------------------

## eval
### llm-eval
Use spacy to extract the noun phrase: 'python -m spacy download en_core_web_sm'