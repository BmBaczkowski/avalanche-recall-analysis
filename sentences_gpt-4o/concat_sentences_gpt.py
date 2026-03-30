import argparse
import pandas as pd
import logging



# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate all dfs.")
    parser.add_argument('filelist', type=str)
    parser.add_argument('outfile', type=str)
    return parser.parse_args()

# Main function
def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_arguments()

    with open(args.filelist, 'r') as file:
        filelist = file.readlines()
    
    dfs = [pd.read_parquet(file.strip(), engine='pyarrow') for file in filelist]
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(args.outfile, index=False)

if __name__ == "__main__":
    main()
