source env/bin/activate

python3 src/initial_classification.py --dataset wikiart --label_cols genre style artist --epochs 20 --batch_size 32 --seed 2830 --log_file_name clf --embedding_folder_name wikiart 
