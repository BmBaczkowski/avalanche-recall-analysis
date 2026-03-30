import argparse
import os
import sys
import logging
import pandas as pd
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('inlist', type=str)
    parser.add_argument('llm_path', type=str)
    parser.add_argument('module_dir', type=str)
    parser.add_argument('outlist', type=str)
    return parser.parse_args()

# Import utility functions
def import_utilities(module_dir):
    sys.path.append(module_dir)
    try:
        from utility_sbert import Embedder
        return Embedder
    except ImportError as e:
        logging.error(f"Error importing module: {e}")
        sys.exit(1)
def load_df(df_file):
    df_file = df_file.strip()
    df = pd.read_csv(df_file, compression='gzip')   
    return df

def embed_text(df, embedder):
    df['mpnet-embedding'] = df['sentence'].apply(lambda x: embedder.get_embedding(x).tolist())
    return df

def get_new_filename(df_file, outdir):
    filename = os.path.basename(df_file).split(".")[0]
    fileout = f"{filename}_embed-mpnet.parquet"
    fileout_path = os.path.join(outdir, fileout)
    return fileout_path
    
def process_file(df_file, embedder, outdir):
    # Load the DataFrame
    df = load_df(df_file)
    
    # Embed the text
    df = embed_text(df, embedder)
    
    # Determine the output path
    fileout_path = get_new_filename(df_file, outdir)

    # Save the DataFrame 
    df.to_parquet(fileout_path, index=False, engine='pyarrow')
    
    # Return the output file path for tracking
    return fileout_path

# Main function
def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_arguments()
    Embedder = import_utilities(args.module_dir)
    embedder = Embedder(args.llm_path)

    with open(args.inlist, 'r') as file:
        list_of_files = file.readlines()

    outdir = os.path.dirname(args.outlist)
    
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list_of_new_files = list(executor.map(lambda f: process_file(f, embedder, outdir), list_of_files))

    # save the list of files
    with open(args.outlist, "w") as file:
        for item in list_of_new_files:
            file.write(item + "\n")

if __name__ == "__main__":
    main()
