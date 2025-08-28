import datasets
import os
import argparse 
from functools import partial

def argument_parser():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--hf_name', type=str, help='Name of HuggingFace dataset to be downloaded')
    parser.add_argument('--local_name', type=str, help='What to call the saved version of the dataset')
    args = vars(parser.parse_args())

    return args


def load_iter_hf_data(dataset_name):

    '''
    Load a dataset from Huggingface

    Args:
        - dataset_name: name of dataset from HuggingFace Hub
    '''

    # load dataset from the hub
    hf_data = datasets.load_dataset(dataset_name, split='train', streaming=True)

    # convert dataset to iterable generator
    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    # convert to dataset to be saved locally
    ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, hf_data), features=hf_data.features)

    return ds

def main():

    # parse args
    args = argument_parser()

    # download specified dataset
    ds = load_iter_hf_data(args['hf_name'])

    # create datafolder
    os.makedirs('data', exist_ok=True)

    # save to disk
    ds.save_to_disk(os.path.join('data', args['local_name']))

if __name__ == '__main__':
    main()

