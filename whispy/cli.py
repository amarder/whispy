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
from .transcribe import transcribe_file, find_default_model, find_whisper_cli, build_whisper_cli
from .recorder import record_audio_until_interrupt, check_audio_devices, test_microphone

console = Console()
app = typer.Typer(
    name="whispy",
    help="Fast speech recognition using whisper.cpp",
    add_completion=False,
)


@app.command(name="transcribe")
def main_transcribe(
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
        whispy audio.wav
        whispy audio.mp3 --model models/ggml-base.en.bin
        whispy audio.wav --language en --output transcript.txt
    """
    
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
                "Please provide a model file with --model or download a model:\n\n"
                "[bold]Option 1: Download to whisper.cpp/models/[/bold]\n"
                "  cd whisper.cpp\n"
                "  sh ./models/download-ggml-model.sh base.en\n\n"
                "[bold]Option 2: Download to models/[/bold]\n"
                "  mkdir -p models\n"
                "  curl -L -o models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin\n\n"
                "Available models: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3"
            )
            raise typer.Exit(1)
    
    # Check if whisper-cli is available
    whisper_cli = find_whisper_cli()
    if not whisper_cli:
        console.print("[yellow]whisper-cli not found. Attempting to build it...[/yellow]")
        if build_whisper_cli():
            whisper_cli = find_whisper_cli()
            if whisper_cli:
                console.print(f"[green]Successfully built whisper-cli: {whisper_cli}[/green]")
            else:
                console.print("[red]Failed to find whisper-cli after build[/red]")
                raise typer.Exit(1)
        else:
            console.print(
                "[red]Error: whisper-cli not found and build failed.[/red]\n"
                "Please build whisper.cpp manually:\n"
                "  cd whisper.cpp\n"
                "  cmake -B build\n"
                "  cmake --build build -j --config Release\n\n"
                "Or ensure whisper-cli is in your PATH."
            )
            raise typer.Exit(1)
    
    if verbose:
        console.print(f"[blue]Using whisper-cli: {whisper_cli}[/blue]")
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


@app.command(name="record-and-transcribe")
def record_and_transcribe(
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
    save_audio: Optional[str] = typer.Option(
        None,
        "--save-audio", "-s",
        help="Save the recorded audio to this file (optional)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    ),
    test_mic: bool = typer.Option(
        False,
        "--test-mic", "-t",
        help="Test microphone before recording"
    ),
) -> None:
    """
    Record audio from microphone and transcribe it
    
    Records audio until you press Ctrl+C, then transcribes the recording.
    
    Examples:
        whispy record-and-transcribe
        whispy record-and-transcribe --model models/ggml-base.en.bin
        whispy record-and-transcribe --language en --output transcript.txt
        whispy record-and-transcribe --save-audio recording.wav
        whispy record-and-transcribe --test-mic
    """
    
    # Test microphone if requested
    if test_mic:
        if not test_microphone():
            console.print("[red]❌ Microphone test failed. Please check your microphone settings.[/red]")
            raise typer.Exit(1)
        console.print("[green]✅ Microphone test passed![/green]")
        return
    
    # Check audio devices
    try:
        device_info = check_audio_devices()
        if verbose:
            console.print(f"[blue]Default input device: {device_info['default_input_info']['name']}[/blue]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not check audio devices: {e}[/yellow]")
    
    # Record audio
    console.print("[bold blue]🎤 Ready to record audio[/bold blue]")
    console.print("[blue]Press Ctrl+C to stop recording and start transcription[/blue]")
    
    try:
        audio_file = record_audio_until_interrupt(
            output_path=save_audio
        )
        
        if verbose:
            console.print(f"[blue]Recorded audio saved to: {audio_file}[/blue]")
        
        # Now transcribe the recorded audio
        console.print("[bold blue]🔄 Starting transcription...[/bold blue]")
        
        # Use the existing transcribe function
        transcribe_audio(
            audio_file=audio_file,
            model=model,
            language=language,
            output=output,
            verbose=verbose
        )
        
        # Clean up temporary file if not saving audio
        if save_audio is None:
            try:
                pathlib.Path(audio_file).unlink()
                if verbose:
                    console.print(f"[blue]Cleaned up temporary file: {audio_file}[/blue]")
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]Warning: Could not clean up temporary file: {e}[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Recording cancelled by user[/yellow]")
        raise typer.Exit(130)  # Standard exit code for Ctrl+C
    except WhispyError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information"""
    console.print(f"whispy version {__version__}")


@app.command()
def build() -> None:
    """Build whisper-cli from whisper.cpp source"""
    console.print("[blue]Building whisper-cli...[/blue]")
    
    if not pathlib.Path("whisper.cpp").exists():
        console.print(
            "[red]Error: whisper.cpp directory not found.[/red]\n"
            "Please clone the whisper.cpp repository first:\n"
            "  git clone https://github.com/ggerganov/whisper.cpp.git"
        )
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Building whisper-cli...", total=None)
        
        if build_whisper_cli():
            progress.update(task, description="Build complete!")
            whisper_cli = find_whisper_cli()
            if whisper_cli:
                console.print(f"[green]Successfully built whisper-cli: {whisper_cli}[/green]")
            else:
                console.print("[red]Build succeeded but whisper-cli not found[/red]")
                raise typer.Exit(1)
        else:
            progress.update(task, description="Build failed!")
            console.print("[red]Failed to build whisper-cli[/red]")
            raise typer.Exit(1)


@app.command()
def info() -> None:
    """Show system information"""
    console.print(f"[bold]whispy version:[/bold] {__version__}")
    
    # Check whisper-cli availability
    whisper_cli = find_whisper_cli()
    if whisper_cli:
        console.print(f"[bold]whisper-cli:[/bold] {whisper_cli}")
    else:
        console.print("[bold]whisper-cli:[/bold] [red]Not found[/red]")
    
    # Try to find available models
    model_path = find_default_model()
    if model_path:
        console.print(f"[bold]Default model:[/bold] {model_path}")
    else:
        console.print("[bold]Default model:[/bold] [red]Not found[/red]")
        
    # Show whisper.cpp directory status
    whisper_cpp_dir = pathlib.Path("whisper.cpp")
    if whisper_cpp_dir.exists():
        console.print(f"[bold]whisper.cpp directory:[/bold] {whisper_cpp_dir.absolute()}")
    else:
        console.print("[bold]whisper.cpp directory:[/bold] [red]Not found[/red]")


def main() -> None:
    """Main entry point for the CLI"""
    app()


if __name__ == "__main__":
    main() 