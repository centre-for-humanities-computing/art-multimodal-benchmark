source env/bin/activate

    parser.add_argument('--dataset', type=str, help='name of HuggingFace dataset') 
    parser.add_argument('--subclasses', nargs='+', help= 'List of classes to run subclassification on')
    parser.add_argument('--subclass_label', type=str, help='whether chosen subclassification task is for genre, styles or artists')
    parser.add_argument('--hidden_layer_size', type=int, help= 'size of hidden layer in clf model')
    parser.add_argument('--epochs', type=int, help="how many epochs to run the model for")
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--model_names', nargs='+', help='list of models to run classification task with')
    parser.add_argument('--savefile_suffix', type=str, help='suffix to add to saved files to identify classification task')
    args = vars(parser.parse_args())


python3 src/subclassifications.py --dataset wikiart_filtered_remapped_FINAL --subclasses peter-paul-rubens rembrandt --subclass_label artist --model_names dinov2-base clip-vit-large-patch14 --savefile_suffix rembrandt_vs_rubens 