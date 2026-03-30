import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import multiprocessing
import logging


# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get the tSNE plot of embeddings.")
    parser.add_argument('infilelist', type=str)
    parser.add_argument('outfilelist', type=str)
    return parser.parse_args()

def plot_tsne(df_file, tsnefile):
    df = pd.read_parquet(df_file)
    vectors = np.array(df['mpnet-embedding'].tolist())

    try :
        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
        vis_dims = tsne.fit_transform(vectors)

        # Create the heatmap
        plt.figure(figsize=(10, 8))

        # Create a scatter plot
        plt.figure(figsize=(10, 6))

        x = np.array([x for x,y in vis_dims])
        y = np.array([y for x,y in vis_dims])

        indx1 = np.where(df['depth'] == 1)[0]
        indx2 = np.where(df['depth'] == 2)[0]
        plt.scatter(x[indx1], y[indx1], alpha=0.3, c='red')
        plt.scatter(x[indx2], y[indx2], alpha=0.3, c='blue')

        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('T-SNE: depth')

        # Save the heatmap to a PDF file
        plt.savefig(tsnefile, format='pdf')

        # Close the plot to free memory
        plt.close()

    except Exception as e:
        # handle exception
        print(f"An error occurred: {e}")
        
# Main function
def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_arguments()
    with open(args.infilelist, 'r') as file:
        filelist = file.readlines()

    tsne_list = [filename.replace(".parquet", "_tsne.pdf") for filename in filelist]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(plot_tsne, [(i.strip(), j.strip()) for i, j in zip(filelist, tsne_list)])

    # save the list of files
    outfilelist = args.outfilelist
    with open(outfilelist, "w") as file:
        for item in tsne_list:
            file.write(item)

if __name__ == "__main__":
    main()
