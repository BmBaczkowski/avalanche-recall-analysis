import argparse
import numpy as np
# from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import pandas as pd
import logging
import joblib
import os


# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train random forests classfier")
    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str)
    return parser.parse_args()

        
# Main function
def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_arguments()
    
    df = pd.read_parquet(args.infile)
    X = np.array(df['mpnet-embedding'].tolist())

    clf = IsolationForest(contamination='auto')
    clf.fit(X)

    joblib.dump(clf, args.outfile)

if __name__ == "__main__":
    main()
