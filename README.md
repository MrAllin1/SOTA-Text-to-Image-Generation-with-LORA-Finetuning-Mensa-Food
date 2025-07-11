# mensa_t2i
SoTA T2I Adapting and Finetuning
--------------------------------------------------------------------------------------------------------------------
To start training run these commands:

chmod +x train_lora_job.sh
sbatch train_lora_job.sh


squeue --me
sacct -j 19429865 --format=JobID,State,Elapsed,MaxRSS
--------------------------------------------------------------------------------------------------------------------
To predict with a certain prompt cd into lora folder and run this:
python predict_lora.py \
  --prompt "Tortellini gefullt mit Ricotta und Spinat Basilikum-Kasesauce Fruhlingszwiebel Tomatenwurfel und geriebener Emmentaler" \
  --num_images 2 \
  --height 256 \
  --width 256 \
  --lora_weights_dir ./lora-adapters-second-train
--------------------------------------------------------------------------------------------------------------------
