from dataclasses import dataclass, fields, field
from typing import Optional, Set
from pathlib import Path
import yaml

@dataclass
class Settings:
    target_lang: str = "English"
    ollama_model: str = "translategemma:4b"
    temperature: float = 0.1
    input_dir: Path = Path("./docs")
    output_dir: Path = Path("./translated_docs")
    cache_name_format: str = ".{}_cache.json"
    hash_algo: str = "SHA1"
    max_parallel_files: int = 2
    translate_file_types: Set[str] = field(default_factory=lambda: {".md"})
    cache_name_foramt: str = f".{target_lang}_cache.json"
    cache_path: Path = Path(output_dir / Path(cache_name_foramt))

    def update_from_dict(self, data: dict):
        """Update settings from a dictionary, ignoring extra keys."""
        valid_keys = {f.name for f in fields(self)}
        for key, value in data.items():
            if key in valid_keys:
                setattr(self, key, value)

settings = Settings()

def load_config(yaml_path: Optional[str] = None, **kwargs):
    """
    The 'Source of Truth' for updating settings.
    Call this in main.py to override defaults.
    """
    global settings
    if yaml_path:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            settings.update_from_dict(data)
    
    if kwargs:
        settings.update_from_dict(kwargs)