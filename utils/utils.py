import argparse
import json
import numpy as np

def cli():
    ### Parse arguments
    parser = argparse.ArgumentParser(
        description="minimal benchmark for sparse lengths sum lookup.")
    
    # model related configuration
    ## If use config, you can ignore this part
    parser.add_argument(
        '-d', "--data-type", type=str, default="float32",
        help="The data type for the input lookup table.")
    parser.add_argument(
        '-e', "--arch-embedding-size", type=str, default="4000000",
        help="Lookup table size.")
    parser.add_argument(
        "--arch-sparse-feature-size", type=int, default=32,
        help="Embedding dimension.")
    parser.add_argument(
        "--num-indices-per-lookup", type=int, default=80,
        help="Sparse feature average lengths, default is ")
    parser.add_argument(
        "--model-name", type=str, default="rm1",
        help="The dlrm type for the input lookup table.")
    parser.add_argument(
        "--num-indices-per-lookup-fixed", type=bool, default=True,
        help="The number of indices per lookup is fixed or not."
    )

    # Dataset related arg
    parser.add_argument(
        "--mini-batch-size", type=int, default=1,
        help="The mini-batch size.")
    # Behavior related arg
    parser.add_argument(
        "--lookup-mode", type=str,default="random",
        help="Random read from storage or from DRAM.")

    # config
    parser.add_argument(
        "--config_file", type=str, default=None)
    parser.add_argument(
        '--gen-table', action="store_true", default=False,
        help="Gen table or not.")


    args = parser.parse_args()

    if args.config_file:
        file_path = args.config_file
        with open(file_path, "r") as f:
            config = json.load(f)
        for key in dict(config).keys():
            type_of = type(getattr(args, key))
            setattr(args, key, type_of(config[key]))

    return args
