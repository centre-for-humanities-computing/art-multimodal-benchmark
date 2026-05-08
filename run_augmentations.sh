source env/bin/activate

#python3 src/wikidata_augmentations.py --dataset wikidata_remapped --model_names dinov2-base
#python3 src/augment_and_save_images.py --data wikidata_remapped

#python3 src/batched_augmentations.py --data wikidata_remapped

#python3 src/extract_augmented_embeddings.py --dataset wikiart_filtered_remapped_FINAL --model_names CLIP-ViT-B-16-DataComp.XL-s13B-b90K CLIP-ViT-L-14-DataComp.XL-s13B-b90K CLIP-ViT-bigG-14-laion2B-39B-b160k dinov2-base dinov2-giant eva02_clip_336 siglip-base-patch16-224 siglip-large-patch16-384 siglip-so400m-patch14-384

#python3 src/extract_augmented_embeddings.py --dataset wikiart_filtered_remapped_FINAL --model_names CLIP-ViT-bigG-14-laion2B-39B-b160k dinov2-base dinov2-giant eva02_clip_336 siglip-base-patch16-224 siglip-large-patch16-384 siglip-so400m-patch14-384


python3 src/classify_augmentations_FINAL.py --dataset wikiart_filtered_remapped_FINAL_AUG_SUBSET --model_names CLIP-ViT-B-16-DataComp.XL-s13B-b90K CLIP-ViT-L-14-DataComp.XL-s13B-b90K CLIP-ViT-bigG-14-laion2B-39B-b160k dinov2-base dinov2-giant eva02_clip_336 siglip-base-patch16-224 siglip-large-patch16-384 siglip-so400m-patch14-384

#python3 src/classify_augmentations_FINAL.py --dataset wikiart_filtered_remapped_FINAL_AUG_SUBSET --model_names CLIP-ViT-B-16-DataComp.XL-s13B-b90K 