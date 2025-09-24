#!/usr/bin/env python3
"""
Ultra-optimized MP4 video downscaler to 480p
Minimal resource usage with maximum speed
Requires: ffmpeg installed on system
"""

import subprocess
import os
import sys
from pathlib import Path
import time

def get_video_info(input_path):
    """Get basic video info without heavy processing"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'csv=p=0',
            '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
            input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split(','))
        return width, height
    except:
        return None, None

def downscale_video(input_path, output_path=None):
    """
    Ultra-fast video downscaling to 480p with minimal resource usage
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_480p{input_path.suffix}"
    else:
        output_path = Path(output_path)
        # Ensure output has .mp4 extension
        if not output_path.suffix:
            output_path = output_path.with_suffix('.mp4')
    
    # Check current resolution
    width, height = get_video_info(input_path)
    if width is None or height is None:
        print(f"Error: Could not read video info from '{input_path}'")
        return False
    
    # Skip if already 480p or smaller
    if height <= 480:
        print(f"Video is already {height}p or smaller, skipping...")
        return True
    
    print(f"Downscaling {width}x{height} → 480p")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    start_time = time.time()
    
    # Ultra-optimized FFmpeg command for speed and minimal resources
    cmd = [
        'ffmpeg', '-y',  # Overwrite output
        '-i', str(input_path),
        
        # Video filters: scale to 480p maintaining aspect ratio
        '-vf', 'scale=-2:480',  # -2 ensures width is divisible by 2
        
        # Ultra-fast encoding settings
        '-c:v', 'libx264',      # Most compatible codec
        '-preset', 'ultrafast', # Fastest encoding preset
        '-crf', '28',           # Balanced quality/size (higher = smaller file)
        
        # Audio: copy without re-encoding to save time
        '-c:a', 'copy',
        
        # Optimization flags
        '-movflags', '+faststart',  # Web-optimized
        '-threads', '0',            # Use all CPU threads
        '-avoid_negative_ts', 'make_zero',
        
        str(output_path)
    ]
    
    # Try with hardware acceleration first, fallback to CPU-only
    try:
        # First attempt with hardware acceleration
        hw_cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'auto',
            '-i', str(input_path)
        ] + cmd[3:]  # Rest of the command
        
        result = subprocess.run(
            hw_cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
    except subprocess.CalledProcessError:
        # Fallback to CPU-only processing
        print("  Hardware acceleration failed, using CPU...")
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error during conversion: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            return False
    
    # Success - calculate stats
    elapsed = time.time() - start_time
    
    # Get file sizes
    input_size = input_path.stat().st_size / (1024*1024)  # MB
    output_size = output_path.stat().st_size / (1024*1024)  # MB
    compression = (1 - output_size/input_size) * 100
    
    print(f"✓ Conversion completed in {elapsed:.1f}s")
    print(f"  Size: {input_size:.1f}MB → {output_size:.1f}MB ({compression:.1f}% smaller)")
    return True

def batch_process(directory, pattern="*.mp4"):
    """Process all MP4 files in a directory"""
    directory = Path(directory)
    files = list(directory.glob(pattern))
    
    if not files:
        print(f"No {pattern} files found in {directory}")
        return
    
    print(f"Found {len(files)} video files to process")
    
    success_count = 0
    total_time = time.time()
    
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
        if downscale_video(file_path):
            success_count += 1
    
    total_time = time.time() - total_time
    print(f"\n{'='*50}")
    print(f"Batch processing completed!")
    print(f"Processed: {success_count}/{len(files)} files")
    print(f"Total time: {total_time:.1f}s")

def main():
    # ===== CONFIGURATION SECTION =====
    # Simply change these paths to your video files:
    
    # OPTION 1: Single file processing
    input_video = "C:/path/to/your/video.mp4"  # ← PUT YOUR VIDEO PATH HERE
    output_video = "C:/path/to/output_480p.mp4"  # Auto-generates output name, or specify: "C:/path/to/output_480p.mp4"

    # OPTION 2: Batch processing (uncomment to use)
    # video_folder = "C:/path/to/your/video/folder"  # ← PUT YOUR FOLDER PATH HERE
    
    # ===== EXECUTION =====
    # Choose ONE option below (comment/uncomment as needed):
    
    # Process single file:
    downscale_video(input_video, output_video)
    
    # Process all MP4s in a folder (uncomment line below and comment line above):
    # batch_process(video_folder)

if __name__ == "__main__":
    main()