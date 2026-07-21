source env/bin/activate

# augment + extract embeddings
python3 src/extract_augmented_embeddings.py --dataset wikiart --model_names CLIP-ViT-B-16-DataComp.XL-s13B-b90K CLIP-ViT-L-14-DataComp.XL-s13B-b90K CLIP-ViT-bigG-14-laion2B-39B-b160k dinov2-base dinov2-giant eva02_clip_336 siglip-base-patch16-224 siglip-large-patch16-384 siglip-so400m-patch14-384 

# classify
python3 src/classify_augmentations.py --dataset wikiart_FINAL_AUG_SUBSET --model_names CLIP-ViT-B-16-DataComp.XL-s13B-b90K CLIP-ViT-L-14-DataComp.XL-s13B-b90K CLIP-ViT-bigG-14-laion2B-39B-b160k dinov2-base dinov2-giant eva02_clip_336 siglip-base-patch16-224 siglip-large-patch16-384 siglip-so400m-patch14-384 --embedding_folder_name wikiart_embeddings