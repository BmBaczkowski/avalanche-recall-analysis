import argparse
import os
import sys
import logging
import joblib
import nltk


# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('recall_file', type=str)
    parser.add_argument('clf_file', type=str)
    parser.add_argument('nltk', type=str)
    parser.add_argument('module_dir', type=str)
    parser.add_argument('outfile', type=str)
    return parser.parse_args()

# Import utility functions
def import_utilities(module_dir):
    sys.path.append(module_dir)
    try:
        from utility_functions import read_text_file
        return read_text_file
    except ImportError as e:
        logging.error(f"Error importing module: {e}")
        sys.exit(1)

# Main function
def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_arguments()
    nltk.data.path.append(args.nltk)

    read_text_file = import_utilities(args.module_dir)


    recall = read_text_file(args.recall_file)
    recall_sent = nltk.sent_tokenize(recall)

    # remove \n

    clf = joblib.load(args.clf_file)

    breakpoint()
    
    with open(args.outfile, "w") as file:
        for item in list_of_files:
            file.write(item + "\n")

if __name__ == "__main__":
    main()
