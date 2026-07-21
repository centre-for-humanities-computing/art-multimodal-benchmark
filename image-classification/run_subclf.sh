source env/bin/activate

# rembrandt vs rubens
python3 src/subclassification.py --dataset wikiart --subclasses Impressionism Expressionism Post_Impressionism --subclass_label style --model_names dinov2-base clip-vit-large-patch14 --savefile_suffix impressionism_versions_CV --cv --embedding_folder_name wikiart_embeddings

# genre vs religious paintings
python3 src/subclassification.py --dataset wikiart --subclasses genre_painting religious_painting --subclass_label genre --model_names CLIP-ViT-B-16-DataComp.XL-s13B-b90K CLIP-ViT-L-14-DataComp.XL-s13B-b90K CLIP-ViT-bigG-14-laion2B-39B-b160k dinov2-base dinov2-giant eva02_clip_336 siglip-base-patch16-224 siglip-large-patch16-384 siglip-so400m-patch14-384 --savefile_suffix genre_vs_religious --cv --embedding_folder_name wikiart_embeddings