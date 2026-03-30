import argparse
import json
import pandas as pd
import logging
import sys
import multiprocessing
import re

# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preproc GPT response to data frame.")
    parser.add_argument('inflie_simplified', type=str)
    parser.add_argument('inflie_detailed', type=str)
    parser.add_argument('module_path', type=str)
    parser.add_argument('outfile', type=str)
    return parser.parse_args()

# Import utility functions
def import_utilities(module_dir):
    sys.path.append(module_dir)
    try:
        from utility_openai import load_response_json
        return load_response_json
    except ImportError as e:
        logging.error(f"Error importing module: {e}")
        sys.exit(1)

def preproc_files(simplified_file, detailed_file, load_response_json):
    simplified_file = simplified_file.strip()
    detailed_file = detailed_file.strip()
    simplified = load_response_json(simplified_file)
    detailed = load_response_json(detailed_file)

    df_sim = pd.DataFrame()
    for i in range(len(simplified['choices'])):
        json_response = simplified['choices'][i]['message']['content']
        gpt_response = json.loads(json_response)
        df_ = pd.json_normalize(gpt_response['response'])
        df_['response_id'] = "{:d}".format(i)
        df_ = df_.explode('sentences')
        df_sim = pd.concat([df_sim, df_], axis=0)
    df_sim = df_sim.reset_index(drop=True)
    df_sim['depth'] = 1
    df_sim = df_sim.rename(columns={"idea_number": "number"})

    df_det = pd.DataFrame()
    for i in range(len(detailed['choices'])):
        json_response = detailed['choices'][i]['message']['content']
        gpt_response = json.loads(json_response)
        df_ = pd.json_normalize(gpt_response['response'])
        df_['response_id'] = "{:d}".format(i)
        df_ = df_.explode('sentences')
        df_det = pd.concat([df_det, df_], axis=0)
    df_det = df_det.reset_index(drop=True)
    df_det['depth'] = 2
    df_det = df_det.rename(columns={"key_point_number": "number"})

    df_concat = pd.concat([df_sim, df_det], ignore_index=True)
    df_concat = df_concat.rename(columns={'sentences': 'sentence'})

    segment_match = re.search(r'segment-(\d+)', simplified_file)
    segment = int(segment_match.group(1))

    df_concat['segment'] = segment

    df_concat = df_concat[
        ["segment",
        "response_id", 
        "number", 
        "depth",
        "sentence"]
    ]

    fileout = simplified_file.replace("_simplified.json.gz", ".csv.gz")
    df_concat.to_csv(fileout, compression="gzip", index=False)
    return fileout

# Main function
def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_arguments()

    load_response_json = import_utilities(args.module_path)

    with open(args.inflie_simplified, 'r') as file:
        list_of_files_simplified = file.readlines()
    with open(args.inflie_detailed, 'r') as file:
        list_of_files_detailed = file.readlines()

    list_of_files_simplified.sort()
    list_of_files_detailed.sort()

    # for file1, file2 in zip(list_of_files_simplified, list_of_files_detailed):
    #     preproc_files(file1, file2, load_response_json) 

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        list_of_new_files = pool.starmap(
            preproc_files, 
            [
                (file1, file2, load_response_json) 
                for file1, file2 
                in zip(list_of_files_simplified, list_of_files_detailed)
            ]
        )
 
    # save the list of files
    with open(args.outfile, "w") as file:
        for item in list_of_new_files:
            file.write(item + "\n")

if __name__ == "__main__":
    main()
