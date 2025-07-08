#!/usr/bin/env python3
"""
Command-line interface for whispy
"""

import pathlib
import sys
from typing import Optional

import typer
import pyperclip
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

from . import __version__, WhispyError
from .transcribe import transcribe_file, find_default_model, find_whisper_cli, build_whisper_cli
from .recorder import record_audio_until_interrupt, check_audio_devices, test_microphone
from .realtime import run_realtime_transcription, test_realtime_setup

console = Console()
app = typer.Typer(
    name="whispy",
    help="🎤 Whispy - A powerful CLI for audio transcription using Whisper",
    add_completion=False,
    no_args_is_help=True,
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
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Transcribing audio...", total=100)
            
            def update_progress(progress_value: float):
                progress.update(task, completed=progress_value * 100)
            
            transcript = transcribe_file(
                audio_path=str(audio_path),
                model_path=model_path,
                language=language,
                progress_callback=update_progress
            )
            
            progress.update(task, description="Transcription complete!", completed=100)
    
    except WhispyError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    
    # Copy to clipboard
    try:
        pyperclip.copy(transcript)
        console.print("📋 [green]Transcript copied to clipboard![/green]")
    except Exception as e:
        if verbose:
            console.print(f"[yellow]Warning: Could not copy to clipboard: {e}[/yellow]")
    
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


@app.command(name="record")
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
    no_volume_indicator: bool = typer.Option(
        False,
        "--no-volume-indicator",
        help="Disable real-time volume indicator during recording"
    ),
) -> None:
    """
    Record audio from microphone and transcribe it
    
    Records audio until you press Ctrl+C, then transcribes the recording.
    Features a real-time volume indicator during recording, progress bar
    during transcription, and automatically copies the result to clipboard.
    
    Examples:
        whispy record-and-transcribe
        whispy record-and-transcribe --model models/ggml-base.en.bin
        whispy record-and-transcribe --language en --output transcript.txt
        whispy record-and-transcribe --save-audio recording.wav
        whispy record-and-transcribe --test-mic
        whispy record-and-transcribe --no-volume-indicator
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
            output_path=save_audio,
            show_volume=not no_volume_indicator  # Volume indicator enabled by default
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


@app.command()
def realtime(
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Path to Whisper model file (auto-detected if not specified)"
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language", "-l",
        help="Language code (e.g., 'en', 'es', 'fr')"
    ),
    chunk_duration: float = typer.Option(
        3.0,
        "--chunk-duration", "-c",
        help="Duration of each audio chunk in seconds (default: 3.0)"
    ),
    overlap_duration: float = typer.Option(
        1.0,
        "--overlap-duration", "-o",
        help="Overlap between chunks in seconds (default: 1.0)"
    ),
    silence_threshold: float = typer.Option(
        0.01,
        "--silence-threshold", "-s",
        help="Voice activity detection threshold (default: 0.01)"
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output", "-f",
        help="Save final transcript to file"
    ),
    show_chunks: bool = typer.Option(
        False,
        "--show-chunks",
        help="Show individual chunk transcripts instead of continuous mode"
    ),
    test_setup: bool = typer.Option(
        False,
        "--test-setup",
        help="Test real-time setup without starting transcription"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    ),
):
    """
    🔴 Real-time transcription from microphone.
    
    Continuously transcribes audio from your microphone in real-time using
    overlapping audio chunks for smooth, continuous transcription.
    
    Examples:
        whispy realtime                    # Start real-time transcription
        whispy realtime --show-chunks      # Show individual chunks
        whispy realtime -c 2.0 -o 0.5     # Custom chunk/overlap timing
        whispy realtime --output live.txt  # Save final transcript
        whispy realtime --test-setup       # Test setup only
    """
    try:
        # Test setup mode
        if test_setup:
            console.print("🔧 Testing real-time transcription setup...")
            success = test_realtime_setup()
            if success:
                console.print("✅ Real-time transcription setup is ready!", style="green")
            else:
                console.print("❌ Real-time transcription setup failed", style="red")
                raise typer.Exit(1)
            return
        
        # Validate parameters
        if chunk_duration <= 0:
            console.print("❌ Chunk duration must be positive", style="red")
            raise typer.Exit(1)
        
        if overlap_duration < 0 or overlap_duration >= chunk_duration:
            console.print("❌ Overlap duration must be between 0 and chunk duration", style="red")
            raise typer.Exit(1)
        
        if silence_threshold < 0 or silence_threshold > 1:
            console.print("❌ Silence threshold must be between 0 and 1", style="red")
            raise typer.Exit(1)
        
        # Show configuration
        config_table = Table(title="🔴 Real-time Transcription Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="white")
        
        config_table.add_row("Model", model or "auto-detected")
        config_table.add_row("Language", language or "auto-detected")
        config_table.add_row("Chunk Duration", f"{chunk_duration}s")
        config_table.add_row("Overlap Duration", f"{overlap_duration}s")
        config_table.add_row("Silence Threshold", f"{silence_threshold}")
        config_table.add_row("Output Mode", "chunks" if show_chunks else "continuous")
        if output_file:
            config_table.add_row("Output File", output_file)
        
        console.print(config_table)
        console.print()
        
        # Run real-time transcription
        final_transcript = run_realtime_transcription(
            model_path=model,
            language=language,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            silence_threshold=silence_threshold,
            output_file=output_file,
            verbose=verbose,
            show_chunks=show_chunks
        )
        
        if final_transcript:
            console.print(f"\n✅ Real-time transcription completed ({len(final_transcript)} characters)")
        else:
            console.print("ℹ️ No audio was transcribed", style="yellow")
            
    except WhispyError as e:
        console.print(f"❌ {e}", style="red")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n🛑 Real-time transcription interrupted", style="yellow")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="red")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


def main() -> None:
    """Main entry point for the CLI"""
    app()


if __name__ == "__main__":
    main() 