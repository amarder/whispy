#!/usr/bin/env python3
"""
Command-line interface for whispy
"""

import pathlib
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__, WhispyError
from .transcribe import transcribe_file, find_default_model

console = Console()
app = typer.Typer(
    name="whispy",
    help="Fast speech recognition using whisper.cpp",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def main_callback(
    ctx: typer.Context,
    audio_file: Optional[str] = typer.Argument(
        None, 
        help="Path to the audio file to transcribe"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Path to the whisper model file. If not provided, will search for a default model."
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language", "-l", 
        help="Language code (e.g., 'en', 'es', 'fr'). If not provided, language will be auto-detected."
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path. If not provided, prints to stdout."
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    ),
) -> None:
    """
    Transcribe an audio file to text using whisper.cpp
    
    Examples:
        whispy audio.wav
        whispy audio.mp3 --model models/ggml-base.en.bin
        whispy audio.wav --language en --output transcript.txt
    """
    
    # If no subcommand is provided and we have an audio file, do transcription
    if ctx.invoked_subcommand is None:
        if not audio_file:
            console.print("[red]Error: Please provide an audio file to transcribe[/red]")
            console.print("Usage: whispy [OPTIONS] AUDIO_FILE")
            console.print("       whispy --help")
            raise typer.Exit(1)
        
        # Call the transcribe function
        return transcribe_audio(audio_file, model, language, output, verbose)


def transcribe_audio(
    audio_file: str,
    model: Optional[str] = None,
    language: Optional[str] = None,
    output: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Internal function to handle transcription"""
    
    # Validate audio file
    audio_path = pathlib.Path(audio_file)
    if not audio_path.exists():
        console.print(f"[red]Error: Audio file not found: {audio_file}[/red]")
        raise typer.Exit(1)
    
    # Find model file
    model_path = None
    if model:
        model_path = pathlib.Path(model)
        if not model_path.exists():
            console.print(f"[red]Error: Model file not found: {model}[/red]")
            raise typer.Exit(1)
        model_path = str(model_path)
    else:
        # Try to find a default model
        model_path = find_default_model()
        if not model_path:
            console.print(
                "[red]Error: No model file specified and no default model found.[/red]\n"
                "Please provide a model file with --model or download a model to one of these locations:\n"
                "  - models/ggml-base.en.bin\n"
                "  - models/ggml-base.bin\n"
                "  - models/ggml-small.en.bin\n"
                "  - models/ggml-small.bin\n"
                "  - models/ggml-tiny.en.bin\n"
                "  - models/ggml-tiny.bin\n\n"
                "You can download models from: https://huggingface.co/ggerganov/whisper.cpp"
            )
            raise typer.Exit(1)
    
    if verbose:
        console.print(f"[blue]Using model: {model_path}[/blue]")
        console.print(f"[blue]Audio file: {audio_file}[/blue]")
        if language:
            console.print(f"[blue]Language: {language}[/blue]")
    
    # Transcribe the audio file
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Transcribing audio...", total=None)
            
            transcript = transcribe_file(
                audio_path=str(audio_path),
                model_path=model_path,
                language=language
            )
            
            progress.update(task, description="Transcription complete!")
    
    except WhispyError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    
    # Output the result
    if output:
        try:
            output_path = pathlib.Path(output)
            output_path.write_text(transcript, encoding='utf-8')
            console.print(f"[green]Transcript saved to: {output_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving to {output}: {e}[/red]")
            raise typer.Exit(1)
    else:
        console.print(transcript)


@app.command()
def transcribe(
    audio_file: str = typer.Argument(
        ..., 
        help="Path to the audio file to transcribe"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Path to the whisper model file. If not provided, will search for a default model."
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language", "-l", 
        help="Language code (e.g., 'en', 'es', 'fr'). If not provided, language will be auto-detected."
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path. If not provided, prints to stdout."
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    ),
) -> None:
    """
    Transcribe an audio file to text using whisper.cpp
    
    Examples:
        whispy transcribe audio.wav
        whispy transcribe audio.mp3 --model models/ggml-base.en.bin
        whispy transcribe audio.wav --language en --output transcript.txt
    """
    return transcribe_audio(audio_file, model, language, output, verbose)


@app.command()
def version() -> None:
    """Show version information"""
    console.print(f"whispy version {__version__}")


@app.command()
def info() -> None:
    """Show system information"""
    from . import is_library_loaded
    
    console.print(f"[bold]whispy version:[/bold] {__version__}")
    console.print(f"[bold]Library loaded:[/bold] {is_library_loaded()}")
    
    # Try to find available models
    model_path = find_default_model()
    if model_path:
        console.print(f"[bold]Default model:[/bold] {model_path}")
    else:
        console.print("[bold]Default model:[/bold] [red]Not found[/red]")


def main() -> None:
    """Main entry point for the CLI"""
    app()


if __name__ == "__main__":
    main() 