import dotenv
import gzip
import json
import logging
import openai
import os
from typing import Any, Dict, List, Optional, Union

# Global Constants and Variables
API_KEY_ENV_VAR = "KEY_API_OPENAI"

def create_message_list(system_prompt: str, user_text: str) -> List[Dict[str, str]]:
    """
    Create a list of messages with system and user roles.

    Args:
        system_prompt (str): The content to be used as the system message.
        user_text (str): The content to be used as the user message.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each representing a message with a role and content.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]

def get_openai_client(api_key: str) -> openai.OpenAI:
    """Initialize and return an OpenAI client."""
    return openai.OpenAI(api_key=api_key)

def get_api_key(env_file: Optional[str] = ".env") -> Optional[str]:
    """
    Load environment variables from a specified .env file and return the API key.

    Args:
        env_file (Optional[str]): Path to the .env file. Defaults to ".env".

    Returns:
        Optional[str]: The API key if loaded successfully, None otherwise.
    """
    if not os.path.exists(env_file):
        logging.error(f".env file not found: {env_file}")
        return None

    try:
        loaded = dotenv.load_dotenv(env_file)
        if not loaded:
            logging.warning(f"No environment variables loaded from: {env_file}")
            return None
        
        api_key = os.getenv(API_KEY_ENV_VAR)
        if not api_key:
            logging.error("API key for OpenAI is missing.")
            return None
        
        logging.info(f"Environment variables loaded successfully from: {env_file}")
        return api_key
    except Exception as e:
        logging.error(f"Error loading .env file: {e}")
        return None

def get_gpt_response(
    system_prompt: str, 
    user_text: str,
    model: str = "gpt-4o-mini-2024-07-18", 
    n_responses: int = 1,  
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    temperature: float = 1.0, 
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0, 
    response_format: Dict[str, str] = {"type": "json_object"}
) -> Optional[Dict[str, Any]]:
    """
    Get a GPT response from the OpenAI API.

    Args:
        system_prompt (str): The system message to initialize the conversation.
        user_text (str): The user's message for the API.
        model (str): Model to use for the completion.
        n_responses (int): Number of responses to generate.
        logprobs (Optional[bool]): Whether to include log probabilities.
        top_logprobs (Optional[int]): Number of top log probabilities to return.
        temperature (float): Sampling temperature.
        presence_penalty (float): Presence penalty parameter.
        frequency_penalty (float): Frequency penalty parameter.
        response_format (Dict[str, str]): Format of the response.

    Returns:
        Optional[Dict[str, Any]]: The API response as a dictionary, or None if an error occurs.
    """
    messages = create_message_list(system_prompt, user_text)
    
    api_key = get_api_key()
    if not api_key:
        logging.error("API key for OpenAI is missing.")
        return None

    try:
        client = get_openai_client(api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logprobs=logprobs,
            n=n_responses,
            top_logprobs=top_logprobs,
            response_format=response_format
        )
        return response
    except openai.error.OpenAIError as e:
        logging.error(f"An OpenAI error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    return None

def read_prompt_file(prompt_file: str) -> str:
    """
    Read the system prompt from a file.

    Args:
        prompt_file (str): Path to the file containing the system prompt.

    Returns:
        str: The content of the system prompt file.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        IOError: If there is an issue reading the file.
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            content = file.read()
        logging.info(f"Successfully read prompt file: {prompt_file}")
        return content
    except FileNotFoundError as e:
        logging.error(f"Prompt file not found: {prompt_file}")
        raise e
    except IOError as e:
        logging.error(f"Error reading prompt file: {prompt_file}")
        raise e

def get_embedding(text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
    """
    Get the embedding for a given text using the OpenAI API.

    Args:
        text (str): The text to embed.
        model (str): The model to use for embedding. Defaults to "text-embedding-3-small".

    Returns:
        Optional[List[float]]: The embedding as a list of floats, or None if an error occurs.
    """
    api_key = get_api_key()
    if not api_key:
        logging.error("API key for OpenAI is missing.")
        return None

    try:
        client = get_openai_client(api_key)
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except openai.error.OpenAIError as e:
        logging.error(f"An OpenAI error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    return None

def load_response_json(file_name: str) -> Union[dict, list, None]:
    """
    Load and return JSON data from a gzip-compressed file.

    Args:
        file_name (str): The path to the gzip-compressed JSON file.

    Returns:
        Union[dict, list, None]: The JSON data parsed from the file, or None if an error occurs.
    """
    try:
        with gzip.open(file_name, "rt", encoding="utf-8") as gz:
            data = json.load(gz)
        return data
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_name}' was not found.")
    except json.JSONDecodeError:
        logging.error(f"Error: The file '{file_name}' could not be decoded as JSON.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    return None

def save_response_json(response: Any, file_name: str) -> None:
    """
    Save JSON data from a response object to a gzip-compressed file.

    Args:
        response (Any): The response object containing the data to be saved.
        file_name (str): The path to the gzip-compressed JSON file.
    """
    try:
        # Ensure that the response can be serialized to JSON
        json_str = response.model_dump_json()
        data = json.loads(json_str)

        with gzip.open(file_name, "wt", encoding="utf-8") as gz:
            json.dump(data, gz, indent=4)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from response: {e}")
    except OSError as e:
        logging.error(f"Error saving to {file_name}: {e}")
    except AttributeError as e:
        logging.error(f"Error accessing model_dump_json method: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
