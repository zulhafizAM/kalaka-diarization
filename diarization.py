from pyannote.audio import Pipeline
from pyannote.core import Segment
import matplotlib.pyplot as plt
import torchaudio
import uuid
import pickle
import random
import os

# Load pipeline
with open("finetuned_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

def process_audio(file_path: str) -> dict:
    waveform, sample_rate = torchaudio.load(file_path)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    fig, ax = plt.subplots(figsize=(12, 4))

    # Assign unique Y positions and colors for each speaker
    speakers = sorted(set(label for _, _, label in diarization.itertracks(yield_label=True)))
    speaker_to_y = {speaker: i for i, speaker in enumerate(speakers)}
    speaker_colors = {
        speaker: "#" + ''.join(random.choices("0123456789ABCDEF", k=6))
        for speaker in speakers
    }

    min_time = float('inf')
    max_time = float('-inf')

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        ax.hlines(
            y=speaker_to_y[speaker],
            xmin=segment.start,
            xmax=segment.end,
            color=speaker_colors[speaker],
            linewidth=10,
        )
        min_time = min(min_time, segment.start)
        max_time = max(max_time, segment.end)

    ax.set_xlim(min_time, max_time)
    ax.set_yticks([])  # Hide y-axis ticks
    ax.set_xlabel("Time (s)")
    ax.set_title("Speaker Diarization")

    # Add legend below plot
    handles = [
        plt.Line2D([0], [0], color=color, lw=6, label=speaker)
        for speaker, color in speaker_colors.items()
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=len(speakers),
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0.2)

    # Generate filenames
    unique_id = uuid.uuid4()
    image_path = f"temp/diarization_{unique_id}.png"
    rttm_path = f"temp/diarization_{unique_id}.rttm"

    # Save figure
    plt.savefig(image_path, bbox_inches='tight')
    plt.close(fig)

    # Save RTTM
    with open(rttm_path, "w") as f:
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            line = (
                f"SPEAKER unknown 1 {segment.start:.3f} "
                f"{segment.duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
            )
            f.write(line)

    return {
        "image_path": image_path,
        "rttm": rttm_path
    }
