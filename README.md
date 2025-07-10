# mensa_t2i
SoTA T2I Adapting and Finetuning


chmod +x train_lora_job.sh
sbatch train_lora_job.sh


squeue --me
sacct -j 19425405 --format=JobID,State,Elapsed,MaxRSS
19425346