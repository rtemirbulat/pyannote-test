import os
import openai
import torch
import torchaudio
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import datetime
import subprocess
import time
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = ""

def safe_transcribe(file_obj, retries=3, base_delay=2):
    """
    Calls openai.Audio.transcribe("whisper-1", file_obj), retrying on APIError.
    """
    for attempt in range(retries):
        try:
            # You can add response_format="verbose_json" to get segments with timestamps
            return openai.Audio.transcribe(
                model="whisper-1",
                file=file_obj,
                response_format="verbose_json"
            )
        except openai.error.APIError as e:
            # If it’s the last attempt, re-raise
            if attempt == retries - 1:
                raise

            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"APIError on attempt {attempt+1}, waiting {wait_time:.1f}s, error: {e}")
            time.sleep(wait_time)

class SpeakerDiarizer:
    def __init__(self, num_speakers=2):
        self.num_speakers = num_speakers
        # Load your speaker embedding model
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def segment_embedding(self, segment, path, duration):
        """
        Given a single segment with start/end times, read that portion of audio
        and compute a speaker embedding vector.
        """
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)

        audio = Audio()
        waveform, sample_rate = audio.crop(path, clip)

        # pass the cropped waveform into the embedding model
        embedding = self.embedding_model(waveform[None])
        return embedding

    def diarize(self, path):
        """
        1) Convert to WAV if needed
        2) Transcribe via Whisper API
        3) Extract embeddings for each segment
        4) Perform speaker clustering
        5) Return a text transcript annotated by speaker
        """

        # If file is not already a WAV, convert it to "audio.wav"
        if not path.endswith('.wav'):
            subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
            path = 'audio.wav'

        # Step 2: Transcribe via Whisper API, with timestamps
        with open(path, "rb") as file_obj:
            result = safe_transcribe(file_obj)

        # The Whisper API returns top-level "text" plus "segments" if we used "verbose_json"
        segments = result["segments"]

        # Get total audio duration
        with contextlib.closing(wave.open(path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        # Step 3: Extract embeddings for each segment
        embeddings = np.zeros(shape=(len(segments), 192))  # ECAPA embedding = 192-d
        for i, segment in enumerate(segments):
            embeddings[i] = self.segment_embedding(segment, path, duration)

        # Remove any NaNs from the embeddings
        embeddings = np.nan_to_num(embeddings)

        # Step 4: Cluster the segments into `num_speakers` speakers
        clustering = AgglomerativeClustering(self.num_speakers).fit(embeddings)
        labels = clustering.labels_

        # Attach speaker labels to each segment
        for i in range(len(segments)):
            segments[i]["speaker"] = "SPEAKER " + str(labels[i] + 1)

        # Step 5: Build a “pretty” transcript with speaker labels and times
        def time_fmt(secs):
            return str(datetime.timedelta(seconds=round(secs)))

        transcript = ""
        for i, segment in enumerate(segments):
            # If this is the first segment or the speaker changed, print a new speaker line
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                transcript += "\n" + segment["speaker"] + " " + time_fmt(segment["start"]) + "\n"
            # Segment text has a preceding space or punctuation sometimes, so strip as needed
            transcript += segment["text"].lstrip() + " "

        return transcript

if __name__ == '__main__':
    diarizer = SpeakerDiarizer(num_speakers=2)
    transcript = diarizer.diarize("pyannote-test/data/sampledata_audiofiles_katiesteve.wav")  # or .wav, etc.
    logger.info(transcript)