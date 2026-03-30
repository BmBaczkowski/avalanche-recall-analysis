import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
import logging

# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get the cosine similarity matrix of embeddings.")
    parser.add_argument('infilelist', type=str)
    parser.add_argument('outfilelist', type=str)
    return parser.parse_args()

def plot_cosine(df_file, cosinefile):
    df = pd.read_parquet(df_file)
    vectors = np.array(df['mpnet-embedding'].tolist())
    sim_mat = cosine_similarity(vectors)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_mat, annot=False, vmin=0, vmax=1)
    plt.title('Cosine similarity')

    # Save the heatmap to a PDF file
    plt.savefig(cosinefile, format='pdf')

    # Close the plot to free memory
    plt.close()
        

# Main function
def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_arguments()
    with open(args.infilelist, 'r') as file:
        filelist = file.readlines()

    cosine_list = [filename.replace(".parquet", "_cosine.pdf") for filename in filelist]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(plot_cosine, [(i.strip(), j.strip()) for i, j in zip(filelist, cosine_list)])

    # save the list of files
    outfilelist = args.outfilelist
    with open(outfilelist, "w") as file:
        for item in cosine_list:
            file.write(item)

if __name__ == "__main__":
    main()