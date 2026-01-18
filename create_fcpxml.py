#!/usr/bin/env python3
"""
Script to convert SRT subtitle files to FCPXML files for DaVinci Resolve.
Processes all SRT files in the subs directory and creates FCPXML files in the kml folder.
"""

import re
import sys
import os
from pathlib import Path


def srt_time_to_seconds(srt_time):
    """Converts SRT timestamp (00:00:00,000) to total seconds."""
    hh, mm, ss_ms = srt_time.split(':')
    ss, ms = ss_ms.split(',')
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def generate_fcpxml(srt_path, output_path, fps=24):
    """Generates an FCPXML file that Resolve imports as Text+ blocks."""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to parse SRT blocks
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.*\n?)+?)(?=\n\d+\n|\Z)')
    matches = pattern.findall(content)

    xml_header = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.8">
    <resources>
        <format id="r1" name="FFVideoFormat1080p{fps}" frameDuration="1/{fps}s"/>
        <effect id="r2" name="Text+" uid=".../Generators.localized/Fusion Generator.localized/Text+.effectid"/>
    </resources>
    <library>
        <event name="Subtitles">
            <project name="Subtitles_to_Blocks">
                <sequence format="r1" duration="3600/1s" tcStart="0/1s" tcFormat="NDF">
                    <spine>
    """

    xml_footer = """
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>"""

    blocks = []
    current_time = 0.0

    for _, start_str, end_str, text in matches:
        start_sec = srt_time_to_seconds(start_str)
        end_sec = srt_time_to_seconds(end_str)
        duration = end_sec - start_sec
        text = text.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Add a gap if there is silence between subtitles
        if start_sec > current_time:
            gap_dur = start_sec - current_time
            blocks.append(f'<gap name="Gap" offset="{int(current_time*1000)}/1000s" duration="{int(gap_dur*1000)}/1000s" start="0/1s"/>')

        # Create the Text+ block
        item = f"""
        <video name="{text[:20]}" offset="{int(start_sec*1000)}/1000s" ref="r2" duration="{int(duration*1000)}/1000s" start="0/1s">
            <param name="Styled Text" key="999/10/10/1" value="{text}"/>
        </video>"""
        blocks.append(item)
        current_time = end_sec

    full_xml = xml_header + "\n".join(blocks) + xml_footer

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_xml)
    print(f"Successfully generated: {output_path}")


def main():
    # Define paths
    subs_dir = Path("/home/michael/Projects/VideoUnderstanding/output_files/frame_level/subs")
    output_dir = Path("/home/michael/Projects/VideoUnderstanding/output_files/frame_level/fcpxml")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all SRT files in the subs directory
    srt_files = list(subs_dir.glob("*.srt"))
    
    if not srt_files:
        print(f"No SRT files found in {subs_dir}")
        sys.exit(1)
    
    print(f"Found {len(srt_files)} SRT file(s) to convert...")
    
    # Process each SRT file
    for srt_path in srt_files:
        # Create output filename (replace .srt with .fcpxml)
        output_filename = srt_path.stem + ".fcpxml"
        output_path = output_dir / output_filename
        
        print(f"\nProcessing: {srt_path.name}")
        generate_fcpxml(srt_path, output_path)
    
    print(f"\nAll FCPXML files created in: {output_dir}")


if __name__ == "__main__":
    main()
