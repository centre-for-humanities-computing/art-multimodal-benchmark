source env/bin/activate

#python3 src/subclassifications.py --dataset wikiart_filtered_remapped_FINAL --subclasses peter-paul-rubens rembrandt --subclass_label artist --model_names dinov2-base clip-vit-large-patch14 --savefile_suffix rembrandt_vs_rubens 

# rembrandt vs rubens
#python3 src/subclf_updated.py --dataset wikiart_filtered_remapped_FINAL --subclasses Impressionism Expressionism Post_Impressionism --subclass_label style --model_names dinov2-base clip-vit-large-patch14 --savefile_suffix impressionism_versions_CV --cv

# genre vs religious paintings
python3 src/subclf_updated.py --dataset wikiart_filtered_remapped_FINAL --subclasses genre_painting religious_painting --subclass_label genre --model_names CLIP-ViT-B-16-DataComp.XL-s13B-b90K CLIP-ViT-L-14-DataComp.XL-s13B-b90K CLIP-ViT-bigG-14-laion2B-39B-b160k dinov2-base dinov2-giant eva02_clip_336 siglip-base-patch16-224 siglip-large-patch16-384 siglip-so400m-patch14-384 --savefile_suffix genre_vs_religious --cv
# impressionism

#python3 src/subclf_updated.py --dataset wikiart_filtered_remapped_FINAL --subclasses genre_painting religious_painting --subclass_label genre --model_names CLIP-ViT-B-16-DataComp.XL-s13B-b90K --savefile_suffix TESTING --cv

#python3 src/subclf_updated.py --dataset wikiart_filtered_remapped_FINAL --subclasses 
######### TESTING ON ILLUSTRATION/SKETCHES

#python3 src/illustration_clf.py --dataset wikiart_filtered_remapped_FINAL --subclasses peter-paul-rubens rembrandt --subclass_label artist --model_names dinov2-base clip-vit-large-patch14 --savefile_suffix rembrandt_vs_rubens_ILLUSTRATIONS
#python3 src/illustration_clf.py --dataset wikiart_filtered_remapped_FINAL --subclasses Baroque Rococo --subclass_label style --model_names dinov2-base clip-vit-large-patch14 --savefile_suffix baroque_vs_rococo_ILLUSTRATIONS

