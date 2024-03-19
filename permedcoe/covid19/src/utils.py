import os
import argparse

##########################################
############ INPUT PARAMETERS ############
##########################################

def create_parser():
    """
    Create argument parser
    """
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("metadata", type=str,
                        help="Input metadata (.tsv)")
    parser.add_argument("model_prefix", type=str,
                        help="Model prefix")
    parser.add_argument("outdir", type=str,
                        help="Output directory")
    parser.add_argument("ko_file", type=str,
                        help="KO file (txt)")
    parser.add_argument("reps", type=int,
                        help="Number of repetitions")
    parser.add_argument("model", type=str,
                        help="Model")
    parser.add_argument("data_folder", type=str,
                        help="Data folder")
    parser.add_argument("max_time", type=int,
                        help="Maximum simulation time")
    return parser


def parse_input_parameters(show=True):
    """
    Parse input parameters
    """
    parser = create_parser()
    args = parser.parse_args()
    if show:
        print()
        print(">>> WELCOME TO THE PILOT WORKFLOW")
        print("> Parameters:")
        print("\t- metadata file: %s" % args.metadata)
        print("\t- model prefix: %s" % args.model_prefix)
        print("\t- output folder: %s" % args.outdir)
        print("\t- ko file: %s" % args.ko_file)
        print("\t- replicates: %s" % str(args.reps))
        print("\t- model: %s" % args.model)
        print("\t- data folder: %s" % args.data_folder)
        print("\t- max time: %d" % args.max_time)
        print("\n")
    return args


################################################
############ CHECK INPUT PARAMETERS ############
################################################

def check_input_parameters(args):
    """
    Check input parameters
    """
    if os.path.exists(args.outdir):
        print("WARNING: the output folder already exists")
    else:
        os.makedirs(args.outdir)
