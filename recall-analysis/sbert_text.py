import argparse
import os
import sys
import logging
import pandas as pd



# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('df_file', type=str)
    parser.add_argument('llm_path', type=str)
    parser.add_argument('module_dir', type=str)
    parser.add_argument('df_file_out', type=str)
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

def embed_text(df, embedder):
    df['mpnet-embedding'] = df['text'].apply(lambda x: embedder.get_embedding(x).tolist())
    return df

# Main function
def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_arguments()
    Embedder = import_utilities(args.module_dir)
    embedder = Embedder(args.llm_path)
    
    df = pd.read_csv(args.df_file) 
    
    # Embed the text
    df = embed_text(df, embedder)

    df.to_parquet(args.df_file_out, index=False, engine='pyarrow')


if __name__ == "__main__":
    main()
