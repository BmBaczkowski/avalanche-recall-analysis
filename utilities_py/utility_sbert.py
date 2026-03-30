from typing import Union, List
from sentence_transformers import SentenceTransformer
import logging

class Embedder:
    def __init__(self, model_path: str):
        """
        Initialize the Embedder with a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained model.
        """
        self.model = SentenceTransformer(model_path)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)  # Configure logging level

    def get_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Get embeddings for the given text or texts.

        Args:
            text (Union[str, List[str]]): Text or list of texts to encode.

        Returns:
            Union[List[float], List[List[float]]]: Embeddings for the input text(s).
        """
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            self.logger.error(f"Error encoding text: {e}")
            return []  # Return an empty list instead of None

# Example usage:
# embedder = Embedder('model/path')
# embeddings = embedder.get_embedding('some text')