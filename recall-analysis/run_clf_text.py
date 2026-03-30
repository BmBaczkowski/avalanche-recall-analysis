import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import joblib
import os


# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train random forests classfier")
    parser.add_argument('infile', type=str)
    parser.add_argument('clf', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('type', type=str)
    return parser.parse_args()

def plot_heatmap(y_prob, pdffile, threshold=.05):
    mask = y_prob < threshold

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        y_prob, 
        annot=False, 
        mask=mask, 
        vmin=threshold, 
        vmax=.7, 
        cmap="viridis",
        linewidths=.5,
        linecolor='black')
    plt.title('Text: clf (topic)')
    plt.xlabel('Topic', fontsize=12)
    plt.ylabel('Text segment', fontsize=12)
    plt.yticks(rotation=0)
    # Save the heatmap to a PDF file
    plt.savefig(pdffile, format='pdf')

    # Close the plot to free memory
    plt.close()

def process_file(df_file, clf):
    df = pd.read_parquet(df_file)
    X = np.array(df['mpnet-embedding'].tolist())
    y_prob = clf.predict_proba(X)
    return y_prob

# Main function
def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_arguments()
    clf = joblib.load(args.clf)
    
    clf_type = args.type

    indir = os.path.dirname(args.infile)
    outdir = os.path.dirname(args.outfile)

    df_file = args.infile
    y_prob = process_file(df_file, clf)
    pdf_file = df_file.replace(".parquet", f"_{clf_type}.pdf").replace(indir, outdir)
    outfile = df_file.replace(".parquet", f"_{clf_type}.joblib").replace(indir, outdir)
    joblib.dump(y_prob, outfile)
    plot_heatmap(y_prob, pdf_file)


if __name__ == "__main__":
    main()
