source env/bin/activate

#python3 src/subclassifications.py --dataset wikiart_filtered_remapped_FINAL --subclasses peter-paul-rubens rembrandt --subclass_label artist --model_names dinov2-base clip-vit-large-patch14 --savefile_suffix rembrandt_vs_rubens 

# rembrandt vs rubens
#python3 src/subclf_updated.py --dataset wikiart_filtered_remapped_FINAL --subclasses Impressionism Expressionism Post_Impressionism --subclass_label style --model_names dinov2-base clip-vit-large-patch14 --savefile_suffix impressionism_versions_CV --cv
#python3 src/subclf_updated.py --dataset wikiart_filtered_remapped_FINAL --subclasses genre_painting religious_painting --subclass_label genre --model_names clip-vit-large-patch14 --savefile_suffix genre_vs_religious_CV --cv
# impressionism

#python3 src/subclf_updated.py --dataset wikiart_filtered_remapped_FINAL --subclasses 
######### TESTING ON ILLUSTRATION/SKETCHES

#python3 src/illustration_clf.py --dataset wikiart_filtered_remapped_FINAL --subclasses peter-paul-rubens rembrandt --subclass_label artist --model_names dinov2-base clip-vit-large-patch14 --savefile_suffix rembrandt_vs_rubens_ILLUSTRATIONS
#python3 src/illustration_clf.py --dataset wikiart_filtered_remapped_FINAL --subclasses Baroque Rococo --subclass_label style --model_names dinov2-base clip-vit-large-patch14 --savefile_suffix baroque_vs_rococo_ILLUSTRATIONS

python3 src/augment_data_clf.py --dataset wikiart_filtered_remapped_FINAL --labels artist --model_names clip-vit-large-patch14