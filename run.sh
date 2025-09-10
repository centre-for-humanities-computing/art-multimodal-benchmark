source env/bin/activate
#python3 src/download_dataset.py --hf_name huggan/wikiart --local_name wikiart
#python3 src/run_mieb.py --leaderboard mieb_leaderboard_no_EVA.csv --n_models 20 --dataset wikiart --label_cols genre style artist --epochs 20 --hidden_layer_size 1200 --batch_size 32 --log_file_name run1

#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export TF_NUM_INTRAOP_THREADS=1
#export TF_NUM_INTEROP_THREADS=1

python3 src/run_single_model.py --model_path laion/CLIP-ViT-bigG-14-laion2B-39B-b160k --dataset wikiart --label_cols genre style artist --epochs 20 --hidden_layer_size 1200 --batch_size 32 --log_file_name TESTER

deactivate