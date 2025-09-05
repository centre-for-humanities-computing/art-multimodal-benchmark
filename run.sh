source env/bin/activate
#python3 src/download_dataset.py --hf_name huggan/wikiart --local_name wikiart
python3 src/run_mieb.py --leaderboard mieb_leaderboard.csv --n_models 20 --dataset wikiart --label_cols genre style artist --epochs 20 --hidden_layer_size 1200 --batch_size 32 --log_file_name run1

deactivate