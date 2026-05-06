"""Custom tokenizer implementation using vocab.json and merges.txt"""

import json
import torch
from typing import List, Dict, Tuple
from pathlib import Path

try:
    from ..llm_sdk import Small_LLM_Model
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from llm_sdk import Small_LLM_Model

class CustomTokenizer:
    """
    Custom tokenizer that mimics Hugging Face transformers without using encode/decode directly.
    Works with BPE (Byte Pair Encoding) vocabulary and merges.
    """

    def __init__(self, vocab_path: str, merges_path: str = None):
        """
        Initialize the custom tokenizer.
        
        Args:
            vocab_path: Path to vocab.json file
            merges_path: Path to merges.txt file (optional for BPE)
        """
        self.vocab = self.load_vocab(vocab_path)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        if merges_path:
            self.merges = self.load_merges(merges_path)
        else:
            self.merges = []
    def load_vocab(self, vocab_path: str) -> Dict[str, int]:
        """Load vocabulary from JSON file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab
    
    def load_merges(self, merges_path: str) -> List[Tuple[str, str]]:
        """Load BPE merges from text file."""
        merges = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
        return merges
    
    def encode_custom(self, text: str) -> torch.Tensor:
        """
        Encode text to token IDs.
        
        TODO: Implementar encoding com BPE
        - Converter texto em bytes/caracteres iniciais
        - Aplicar merges sucessivos até convergência
        - Retornar tensor com shape [1, num_tokens]
        
        Args:
            text: String to encode
            
        Returns:
            torch.Tensor: Shape [1, num_tokens] com token IDs
        """
        # Placeholder
        ids = []
        # TODO: Implementar aqui
        return torch.tensor([ids], dtype=torch.long)
    
    def decode_custom(self, ids: torch.Tensor | List[int]) -> str:
        """
        Decode token IDs to text.
        
        TODO: Implementar decoding
        - Converter cada ID para string usando reverse_vocab
        - Concatenar as strings
        - Lidar com special tokens se necessário
        
        Args:
            ids: Token IDs (Tensor or list)
            
        Returns:
            str: Decoded text
        """
        # Converter Tensor para lista se necessário
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            # Se é 2D, pegar primeiro elemento
            if isinstance(ids[0], list):
                ids = ids[0]
        
        # TODO: Implementar aqui
        tokens = []

        for id in ids:
            token = self.reverse_vocab.get(id, "")
            
            if token.startswith("Ġ"):
                token = " " + token[1:]

            print(f"ID: {id} -> Token: '{token}'")
            tokens.append(token)
        print(tokens)
        return "".join(tokens)


# Exemplo de uso:
if __name__ == "__main__":
    # Carregar paths do modelo
    # tokenizer = CustomTokenizer(vocab_path="path/to/vocab.json", merges_path="path/to/merges.txt")
    # ids = tokenizer.encode_custom("Hello world")
    # text = tokenizer.decode_custom(ids)
    a = Small_LLM_Model()
    vocab = a.get_path_to_vocab_file()
    merges_path = a.get_path_to_merges_file()
    c = CustomTokenizer(vocab_path=vocab, merges_path=merges_path)
    d = a.get_path_to_tokenizer_file()
    with open(d) as f:
        n = json.load(f)
    ad = a.encode("Hello world")
    print(f"shape {ad.shape}")
    print(c.decode_custom(ad[0]))
    # print(n["pre_tokenizer"])