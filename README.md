# mensa_t2i
SoTA T2I Adapting and Finetuning


chmod +x train_lora_job.sh
sbatch train_lora_job.sh


squeue --me
sacct -j 19425405 --format=JobID,State,Elapsed,MaxRSS
19425346


Prompt for predicting with the finetuned model

python predict_lora.py \
  --prompt "Tortellini gefüllt mit Ricotta und Spinat Basilikum-Käsesauce Frühlingszwiebel Tomatenwürfel und geriebener Emmentaler" \
  --num_images 4 \
  --height 256 \
  --width 256