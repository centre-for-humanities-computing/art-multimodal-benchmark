source env/bin/activate

# Path to the TXT file
MODEL_LIST="last_models.txt"

# Read each line (i.e., model name) and run the Python script
while IFS= read -r model_name; do

    safe_name="${model_name//\//_}"

    python3 src/run_mieb.py --model_path "$model_name" --dataset wikiart --label_cols genre style artist --epochs 20 --hidden_layer_size 3000 --batch_size 32 --log_file_name "$safe_name"

done < "$MODEL_LIST"

deactivate