source env/bin/activate

python3 src/classification_wikidata.py --dataset wikidata_remapped --model_names CLIP-ViT-B-16-DataComp.XL-s13B-b90K CLIP-ViT-L-14-DataComp.XL-s13B-b90K CLIP-ViT-bigG-14-laion2B-39B-b160k dinov2-base dinov2-giant eva02_clip_336 siglip-base-patch16-224 siglip-large-patch16-384 siglip-so400m-patch14-384
