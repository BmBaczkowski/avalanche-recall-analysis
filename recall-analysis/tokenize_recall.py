import argparse
import os
import sys
import logging
import pandas as pd
import spacy


# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=str)
    parser.add_argument('spacy', type=str)
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
    read_text_file = import_utilities(args.module_dir)
    nlp = spacy.load(args.spacy)

    # List all files in the directory
    all_files = os.listdir(args.datadir)
    full_paths = [os.path.join(args.datadir, file) for file in all_files]

    # Filter out the .txt files
    txt_files = [file for file in full_paths if file.endswith('.txt')]
    outdir = os.path.dirname(args.outfile)

    df_files = []
    for txt_file in txt_files:
        recall = read_text_file(txt_file)
        recall = recall.replace('\n', ' ')
        doc = nlp(recall)
        sentences = [sent.text for sent in doc.sents]
        id = [i for i in range(len(sentences))]
        fileout = txt_file.replace(".txt", ".csv.gz")
        fileout = fileout.replace(args.datadir, outdir)
        df_files.append(fileout)
        df = pd.DataFrame({"id": id, "sentence": sentences})
        df.to_csv(fileout, compression="gzip", index=False)

    with open(args.outfile, "w") as file:
        for item in df_files:
            file.write(item + "\n")

if __name__ == "__main__":
    main()
