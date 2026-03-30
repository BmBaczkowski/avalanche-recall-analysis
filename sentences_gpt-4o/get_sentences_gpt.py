import argparse
import os
import sys
import logging


# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get related sentences from GPT-4 model.")
    parser.add_argument('prompt_file', type=str, help="Path to the file containing the system prompt")
    parser.add_argument('text_file', type=str, help="Path to the file with the text on avalanches")
    parser.add_argument('module_dir', type=str, help="Path to the directory containing the utility module")
    parser.add_argument('outfile', type=str, help="Path to the output file")
    parser.add_argument('version', type=str, choices=['simplified', 'detailed'])
    return parser.parse_args()

# Import utility functions
def import_utilities(module_dir):
    sys.path.append(module_dir)
    try:
        from utility_functions import read_text_file, segment_text_avalanches
        from utility_openai import get_gpt_response, read_prompt_file, save_response_json
        return (
            get_gpt_response,
            read_prompt_file,
            read_text_file,
            save_response_json,
            segment_text_avalanches
        )
    except ImportError as e:
        logging.error(f"Error importing module: {e}")
        sys.exit(1)

# Main function
def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_arguments()

    (
    get_gpt_response,
    read_prompt_file,
    read_text_file,
    save_response_json,
    segment_text_avalanches
    ) = import_utilities(args.module_dir)
    
    prompt_system = read_prompt_file(args.prompt_file)
    text_avalanches = read_text_file(args.text_file)
    segments = segment_text_avalanches(text_avalanches)
    
    outdir = os.path.dirname(args.outfile)

    list_of_files = []
    for segment in segments:
        gpt_response = get_gpt_response(prompt_system, segment[1], n_responses=20)
        fileout_path = os.path.join(outdir, f"gpt-4o_segment-{segment[0]:02d}_{args.version}.json.gz")
        list_of_files.append(fileout_path)
        save_response_json(gpt_response, fileout_path)
    
    with open(args.outfile, "w") as file:
        for item in list_of_files:
            file.write(item + "\n")

if __name__ == "__main__":
    main()
