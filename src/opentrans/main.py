import click
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from . import config
from .translator import Translator


@click.command()
@click.argument(
    'input_dir',
    type=click.Path(exists=True, path_type=Path)
)
@click.argument(
    'output_dir',
    type=click.Path(path_type=Path)
)
@click.argument(
    'config_path',
    type=click.Path(path_type=Path)
)
@click.option('--model', '-m', default='translategemma:4b', help='Ollama model name.')
@click.option('--lang', '-l', default='Chinese', help='Target translation language.')
@click.option(
    '--input_dir', '-i',
    type=click.Path(exists=True, path_type=Path),
    help='Directory containing the documents to translate.'
)
@click.option(
    '--output_dir', '-o',
    type=click.Path(path_type=Path),
    help='Directory where translated files will be saved.'
)
@click.option(
    '--config_path', '-c',
    type=click.Path(path_type=Path),
    help='Custom path to a config.yaml settings file.'
)

def main(input_dir, output_dir, config_path, model, lang):
    """
    OpenTrans: Batch translate documentation using local Ollama LLMs.

    A specialized tool for translating Markdown and LaTeX files while preserving 
    input directory structures and protecting technical syntax (code, math, and links).

    Key Features:
    - Local-First: Powered by Ollama for private, cost-free translation.
    - Syntax Shield: Automatically protects backticks, LaTeX, and URLs from translation.
    - Parallel Processing: High-speed batch handling for large documentation sets.
    - Structure Preservation: Mirror your source directory exactly in the output.
    """
    
    if config_path:
      config.load_config(config_path)

    if input_dir:
      config.settings.input_dir = input_dir.resolve()

    if output_dir:
      config.settings.output_dir = output_dir.resolve()
    
    if model:
      config.settings.ollama_model = model
    
    if lang:
      config.settings.target_lang = lang

    if not config.settings.input_dir.exists():
      click.echo(
        f"Error: Input directory {config.settings.input_dir} does not exist.", err=True)
      return

    all_files = [f for f in config.settings.input_dir.rglob('*') if f.is_file()]

    click.echo(f"Target Language: {config.settings.target_lang}")
    click.echo(f"Input:  {config.settings.input_dir}")
    click.echo(f"Output: {config.settings.output_dir}")
    click.echo(f"Using Model: {config.settings.ollama_model}")

    translator = Translator(output_dir / config.settings.cache_filename)
    with ThreadPoolExecutor(max_workers=config.settings.max_parallel_files) as executor:
      list(tqdm(
        executor.map(
          lambda f: translator.process_file(
              f, config.settings.input_dir, config.settings.output_dir),
          all_files
        ),
        total=len(all_files),
        desc="Translating Files",
        unit="file"
      ))

    click.echo("\nComplete.")


if __name__ == '__main__':
    main()
