import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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
    y = np.array(df['depth'].tolist()) - 1
    # Split the data into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=424
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    joblib.dump(clf, args.outfile)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    y_prob_max = np.max(y_prob, axis=1)
    labels = np.unique(y_test)
    y_prob_median = [np.median(y_prob_max[y_test == label]) for label in labels]

    # Calculate metrics
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    report_df['max_prob_median'] = y_prob_median + [0, 0, 0]
    report_str = report_df.to_string()

    report_file = args.outfile.replace(".joblib", "_report.txt")
    # Write the output to a file
    with open(report_file, 'w') as file:
        file.write(report_str)


if __name__ == "__main__":
    main()
