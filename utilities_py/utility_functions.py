from typing import List, Tuple


def read_text_file(text_file: str) -> str:
    """
    Read the content of a text file and return it as a string.

    Args:
        text_file (str): The path to the text file to be read.

    Returns:
        str: The content of the text file.
    """
    try:
        with open(text_file, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {text_file} does not exist.")
    except IOError as e:
        raise IOError(f"An error occurred while reading the file: {e}")


def segment_text_avalanches(text: str) -> List[Tuple[int, str]]:
    """
    Replace escaped single quotes in the text, split the content based on '---',
    and return a list of enumerated segments.

    Args:
        text (str): The input text to be processed.

    Returns:
        List[Tuple[int, str]]: A list of tuples, each containing an index and a segment of text.
    """
    # Replace escaped single quotes with normal single quotes
    text = text.replace("\\'", "'")

    # Split the content based on '---' and create a tuple with enumerated parts
    segments = [(index, part.strip()) for index, part in enumerate(text.split('---'))]
    
    return segments
