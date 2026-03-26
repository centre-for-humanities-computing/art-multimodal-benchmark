source env/bin/activate

MODEL_LIST="laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K laion/CLIP-ViT-bigG-14-laion2B-39B-b160k facebook/dinov2-base facebook/dinov2-giant __/eva02_clip_336 google/siglip-base-patch16-224 google/siglip-large-patch16-384 google/siglip-so400m-patch14-384"

python3 src/embeddings_wikidata.py --models $MODEL_LIST --dataset wikidata