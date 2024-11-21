
# Chapter No: 7 **Introduction to MIR**
## What Is MIR?

Have you ever wondered how Spotify suggests music you might like? Or how Shazam can identify a song playing in a noisy cafe? These feats of audio magic are powered by **Music Information Retrieval (MIR)**, a field dedicated to extracting meaningful information from music audio. In this chapter, we'll delve into the world of MIR, exploring its core concepts and applications, demonstrating how you can leverage Python to analyze and understand music in new ways. Think of it as teaching your programs to "listen" to and "understand" music, just like we do.  We'll start with simple examples and gradually build up to more complex tasks.

### Definition and Scope

Music Information Retrieval (MIR) is a multidisciplinary field that sits at the intersection of computer science, signal processing, and musicology.  Its primary goal is to develop algorithms and systems that can automatically analyze, understand, and organize digital music. Imagine having a program that can listen to a song and tell you its genre, the instruments being played, or even the mood it evokes. That's the power of MIR.  It's not just about identifying songs; it's about extracting a wealth of information embedded within the audio itself.

### Historical Overview

The roots of MIR can be traced back to the mid-20th century, with early attempts to analyze musical scores using computers. However, the field truly took off in the late 1990s with the rise of digital music and increased computational power. Early MIR research focused on tasks like automatic music transcription and genre classification.  As the field matured, new challenges emerged, including music recommendation, source separation, and music generation.

### Current State of the Field

Today, MIR is a vibrant and rapidly evolving field, fueled by the ever-growing availability of digital music and advancements in machine learning and deep learning. Open-source libraries like Librosa and Essentia have made MIR tools and techniques accessible to a broader audience.  With ongoing research and development, MIR continues to push the boundaries of what's possible with music and technology.

### Applications

#### Music Streaming Services

MIR is the backbone of music streaming services like Spotify and Apple Music. It powers features like music recommendation, playlist generation, and automatic music tagging.

#### Music Production

Music producers use MIR tools for tasks like automatic music transcription, source separation, and beat tracking.  These tools can help streamline the production process and unlock new creative possibilities.

#### Music Education

MIR technologies are being incorporated into music education platforms to provide personalized feedback and interactive learning experiences.  Imagine a program that can listen to a student playing an instrument and provide real-time feedback on their performance.

#### Research

MIR is a rich area of research, with ongoing work exploring new algorithms, techniques, and applications. Researchers are constantly pushing the boundaries of what's possible with music and technology.


```python
import librosa
import numpy as np

def get_tempo(audio_file):
    """
    Estimates the tempo of an audio file using Librosa.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        float: Estimated tempo in beats per minute (BPM).
        None: If tempo estimation fails.
    """
    try:
        y, sr = librosa.load(audio_file)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return tempo[0]
    except Exception as e: # Catches potential errors during file loading or tempo estimation
        print(f"Error estimating tempo: {e}")
        return None



# Example usage:
audio_file = "audio.wav"  # Replace with your audio file
tempo = get_tempo(audio_file)

if tempo is not None:
    print(f"Estimated tempo: {tempo:.2f} BPM")




```

**Common Pitfalls:**

* **Incorrect File Paths:** Ensure the `audio_file` path is correct.  Use absolute paths if necessary.
* **Unsupported File Formats:** Librosa supports many formats, but not all. Check the documentation for compatibility.
* **Computational Cost:**  Tempo estimation can be computationally intensive for long audio files. Consider using smaller segments for faster processing.

**Best Practices:**

* **Preprocessing:** Consider preprocessing the audio (e.g., noise reduction) before tempo estimation.
* **Parameter Tuning:** Librosa's tempo estimation function has parameters that can be adjusted. Experiment to find optimal settings for your specific use case.


**Practice Exercises:**

1. **Tempo Variation:**  Analyze the tempo of different genres of music. Are there noticeable differences?
2. **Error Handling:** Modify the `get_tempo` function to handle different types of errors more gracefully.
3. **Real-time Tempo:** Explore how to estimate tempo in real-time using libraries like PyAudio.
## Common MIR Tasks

This section explores common tasks in Music Information Retrieval (MIR), demonstrating how we can use Python to analyze, understand, and even generate music. Imagine building a music streaming service that can automatically categorize songs by genre, detect the mood of a piece, or even generate personalized playlists. These are the kinds of real-world applications MIR empowers us to create.  We'll start with some fundamental tasks and progressively introduce more complex ones, always keeping a programmer's perspective in mind.

### Music Generation

Imagine creating music automatically – not just simple melodies, but complex pieces with varying instruments and rhythms.  This is the realm of music generation.  While a full exploration is beyond our scope here, we'll touch on some basic concepts and demonstrate a simple example using MIDI.

**Concept Explanation**: Music generation algorithms range from rule-based systems (defining musical structures explicitly) to complex deep learning models that learn patterns from vast musical datasets.

**Code Example**:

```python
import pretty_midi

# Create a PrettyMIDI object
midi_data = pretty_midi.PrettyMIDI()

# Create an instrument instance (piano)
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano = pretty_midi.Instrument(program=piano_program)

# Create a note
note = pretty_midi.Note(velocity=100, pitch=60, start=0, end=0.5) # Middle C
piano.notes.append(note)


# Add the instrument to the MIDI data
midi_data.instruments.append(piano)

# Write out the MIDI data
midi_data.write('generated_c_note.mid')
```

**Common Pitfalls**:  MIDI generation can be complex. Ensure you understand the relationship between note numbers, pitches, and durations.

**Practice Suggestions**:
1.  Modify the code to generate a simple scale (e.g., C major).
2.  Experiment with different instruments and note velocities.

### Music Transcription

Music transcription is the process of converting audio recordings into symbolic representations, like sheet music or MIDI files. This is a challenging task due to the complexity of music, including polyphony (multiple notes played simultaneously), variations in tempo and dynamics, and the presence of noise in recordings.

**Concept Explanation**:  Think of it like converting speech to text, but for music. We analyze the audio to identify notes, rhythms, and other musical elements.

**Code Example**: Basic transcription with Librosa (simplified example, full transcription requires more sophisticated techniques):

```python
import librosa
import numpy as np

y, sr = librosa.load('simple_melody.wav') # Load a simple melody
onset_frames = librosa.onset.onset_detect(y=y, sr=sr) # Detect note onsets
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

print(f"Detected onsets at times: {onset_times}")


```

**Common Pitfalls**:  Accuracy can be heavily affected by audio quality and the complexity of the music.  Simple techniques might struggle with polyphonic music.

**Practice Suggestions**:
1. Try transcribing different types of audio (monophonic vs. polyphonic).
2. Explore advanced transcription libraries like `jams`.

### Music Recommendation

Ever wonder how music streaming services suggest songs you might like? Music recommendation uses MIR techniques to analyze your listening habits and suggest similar music.

**Concept Explanation**: We can represent songs as vectors of features (genre, tempo, mood, etc.).  Similar songs have similar feature vectors.

**Code Example**:  (Conceptual - requires pre-computed features and a similarity metric).

```python
import numpy as np

def recommend_songs(user_profile, song_library):
    """Recommends songs based on user profile and a song library.

    Args:
        user_profile: Feature vector representing user preferences.
        song_library: A dictionary of song IDs to feature vectors.

    Returns:
        A list of recommended song IDs.
    """

    recommendations = []
    for song_id, features in song_library.items():
        similarity = np.dot(user_profile, features) / (np.linalg.norm(user_profile) * np.linalg.norm(features)) # Cosine similarity
        recommendations.append((song_id, similarity))

    recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity
    return [song_id for song_id, _ in recommendations[:10]] # Return top 10


# Example Usage (conceptual):
user_profile = np.array([0.8, 0.2, 0.5])  # Example user profile
song_library = {
    'song1': np.array([0.9, 0.1, 0.4]),
    'song2': np.array([0.2, 0.7, 0.1]),
}
recommended_songs = recommend_songs(user_profile, song_library)
print(recommended_songs)


```

**Common Pitfalls**: Defining a good "similarity" measure is crucial. The choice of features greatly impacts recommendation quality.

**Practice Suggestions**:  Explore different similarity metrics (cosine similarity, Euclidean distance).  Think about how user profiles could be built.


### Audio Classification

Audio classification is a fundamental task in MIR, where we categorize audio segments based on their content. This section will cover three key areas: genre classification, instrument recognition, and mood detection.

#### Genre Classification

Genre classification is a common MIR task where we automatically assign genre labels to musical pieces. Think about how music streaming services categorize songs into rock, pop, jazz, etc.

**Concept Explanation**: We extract features from audio and train a model to recognize patterns associated with different genres.

**Code Example**: (Conceptual example - assumes pre-extracted features and a trained classifier).

```python
import librosa
import numpy as np

# Dummy classifier (for illustration, replace with a trained one)
def mock_classifier(features):
    return "Pop"

# Load the audio file

y, sr = librosa.load("song.wav")

# Extract features (example: MFCCs, tempo, spectral centroid, etc.)
# ... (Feature extraction code, covered in previous sections)

# Classify the genre
genre = mock_classifier(features)

print("Predicted genre:", genre)
```

**Common Pitfalls**: Genre boundaries can be blurry, and musical styles evolve. Ensure your dataset is diverse and representative.

**Practice Suggestions**:  Try classifying different genres.  Think about what features might be useful (tempo, harmony, instrumentation).


#### Instrument Recognition

Instrument recognition aims to identify which instruments are present in an audio recording. Imagine automatically tagging songs based on the instruments they contain (e.g., "piano," "guitar," "drums").

**Concept Explanation**:  Similar to genre classification, we analyze the audio to extract features that are characteristic of different instruments.

**Code Example**: (Conceptual - requires pre-trained models and appropriate feature extraction).


```python
import librosa

y, sr = librosa.load("instrumental_piece.wav")

# Feature extraction and model prediction (advanced topic)
# Assume a function 'predict_instruments' does this
instruments = predict_instruments(y, sr) # A function that calls a model for prediction
print(f"Detected instruments: {instruments}")


```

**Common Pitfalls**:  Distinguishing between similar instruments can be challenging. The quality of the recording and the mix can impact accuracy.

**Practice Suggestions**:  Explore datasets of isolated instrument sounds. Think about how features like timbre and frequency content could be used.

#### Mood Detection

Mood detection automatically labels music with emotional tags like "happy," "sad," "energetic," or "calm."  This can be used to create mood-based playlists or analyze the emotional impact of music.

**Concept Explanation**: We extract features related to tempo, harmony, and dynamics, which are often correlated with perceived mood.

**Code Example**: (Conceptual - requires a trained model and appropriate feature extraction).

```python
import librosa
import numpy as np


# Dummy model for illustration
def predict_mood(features):
    if features[0] > 0.5: # Dummy logic
        return "Happy"
    else:
        return "Sad"

y, sr = librosa.load("song.wav")

# Extract features: tempo, spectral characteristics, etc.
# Placeholder. In practice use librosa etc for feature extraction

features = np.array([0.7]) # Fake features

mood = predict_mood(features)

print(f"Detected mood: {mood}")

```


**Common Pitfalls**: Mood is subjective.  Training data needs to reflect this subjectivity, and evaluation can be complex.

**Practice Suggestions**: Explore datasets annotated with mood labels.  Think about how features like tempo and harmony contribute to perceived mood.

### Audio Analysis

Audio analysis delves into extracting low-level information from audio signals. This section will cover pitch detection, beat tracking, and chord recognition.


#### Pitch Detection

Pitch detection identifies the fundamental frequency of a sound, essentially telling us which note is being played. This is fundamental for many MIR tasks.

**Concept Explanation**:  We analyze the frequency content of the audio to determine the dominant frequency, which corresponds to the perceived pitch.

**Code Example**:  Basic pitch detection with Librosa:
```python
import librosa
import numpy as np

y, sr = librosa.load('monophonic_note.wav') # Load a monophonic audio file (single note)
pitches, magnitudes = librosa.piptrack(y=y, sr=sr) # Estimate pitch

# Get the most prominent pitch at each frame
estimated_pitches = np.argmax(magnitudes, axis=0)

# Convert pitch indices to frequencies
frequencies = librosa.midi_to_hz(estimated_pitches)

print(f"Estimated frequencies: {frequencies}")

```

**Common Pitfalls**:  Pitch detection can be challenging with polyphonic music or noisy recordings.

**Practice Suggestions**: Experiment with different audio recordings containing single notes and more complex musical phrases.

#### Beat Tracking

Beat tracking identifies the rhythmic pulse of music, essentially finding the "beat" you would tap your foot to. This is crucial for applications like music synchronization and rhythm analysis.

**Concept Explanation**:  We analyze the audio's temporal structure to detect recurring patterns that correspond to the beat.

**Code Example**: Basic beat tracking with Librosa:

```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('song_with_beat.wav')
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

onset_env = librosa.onset.onset_strength(y, sr=sr)
times = librosa.times_like(onset_env, sr=sr)
plt.plot(times, librosa.util.normalize(onset_env), label='Onset Strength')

beat_times = librosa.frames_to_time(beats, sr=sr)
plt.vlines(beat_times, 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')

plt.legend()
plt.xlabel("Time (seconds)")
plt.show()


print(f"Estimated tempo: {tempo} BPM")
print(f"Beat frames: {beats}")

```

**Common Pitfalls**:  Complex rhythms and tempo changes can make beat tracking difficult.

**Practice Suggestions**: Try beat tracking different genres of music. Compare the results with your own perception of the beat.


#### Chord Recognition

Chord recognition identifies the chords played in a piece of music.  This is important for music analysis, transcription, and generation.

**Concept Explanation**: We analyze the harmonic content of the audio to determine which combinations of notes (chords) are present.

**Code Example**: (Conceptual. Requires specialized libraries and models, which are more advanced).

```python
import librosa

y, sr = librosa.load('song_with_chords.wav')
# Chord recognition using a library like 'madmom' or 'chordrec' (advanced).
# ... (chord recognition logic using external library)


# Assume  'recognize_chords' refers to a function that returns chords
chords = recognize_chords(y,sr)
print(f"Recognized chords: {chords}")

```

**Common Pitfalls**:  Chord recognition is a complex task, particularly with complex harmonies and overlapping instruments.

**Practice Suggestions**: Explore dedicated chord recognition libraries.  Start with simple chord progressions.
## Basic MIR Pipeline

### Introduction

Imagine you're building a music streaming service.  You want to automatically tag songs with genres, recommend similar artists, or even create personalized playlists based on mood. These tasks, and many more, fall under the umbrella of Music Information Retrieval (MIR). To accomplish them, we need a structured approach: a pipeline that takes raw audio as input and outputs meaningful information. This section introduces the basic MIR pipeline, a fundamental concept in understanding how we extract knowledge and insights from music data. Think of it as an assembly line for processing audio, where each station performs a specific task.  This structured approach makes MIR tasks manageable and efficient.

This section will walk you through the different stages of a basic MIR pipeline, explaining the purpose of each component and demonstrating how they work together. We'll use practical Python examples to illustrate the concepts, focusing on simplicity and clarity for programmers new to audio processing. By the end, you'll have a solid grasp of the MIR workflow and be ready to tackle more complex projects.

### Overview of Components

The basic MIR pipeline consists of several key components, each playing a crucial role in transforming raw audio into usable data.

#### Audio Input

This is where the journey begins.  Your audio input could be anything from a local MP3 file to a live audio stream. In Python, libraries like `librosa` and `soundfile`  handle loading audio into a format suitable for processing:

```python
import librosa

# Load an audio file
audio_data, sample_rate = librosa.load("audio.mp3")

# 'audio_data' is a NumPy array containing the audio samples
# 'sample_rate' is the number of samples per second
```

#### Preprocessing

Raw audio often contains noise, silence, or unwanted artifacts. Preprocessing steps aim to clean and prepare the audio for analysis. Common techniques include:

* **Resampling**: Changing the sample rate (e.g., from 44.1kHz to 22.05kHz).
* **Normalization**: Adjusting the volume to a standard range.
* **Silence Removal**: Trimming leading and trailing silence.

```python
import librosa

audio_data, sample_rate = librosa.load("audio.mp3", sr=22050)  # Resample to 22.05kHz
audio_data = librosa.util.normalize(audio_data)  # Normalize audio
```

#### Feature Extraction

This stage transforms the audio waveform into a set of numerical features that represent its characteristics.  These features could capture aspects like rhythm, timbre, or pitch.  Examples include:

* **MFCCs (Mel-Frequency Cepstral Coefficients)**:  Represent the spectral envelope of the sound.
* **Spectral Centroid**:  Indicates the "brightness" of the sound.
* **Tempo**: The speed of the music.

```python
import librosa.feature

mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=12)
spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
```

#### Analysis/Classification

Here, we apply algorithms to analyze the extracted features. This could involve:

* **Genre Classification**:  Training a machine learning model to predict the genre of a song.
* **Mood Detection**: Determining the emotional tone of a piece of music.
* **Similarity Search**: Finding similar songs based on their features.

```python
# Example: Simple genre classification (using a placeholder model)
# Assume 'model' is a pre-trained classifier and 'genres' is a list of genre labels
# This is a simplified example; real-world classification requires more complex models and training data
predicted_genre = model.predict(mfccs.T)[0]  # 'model' and 'genres' are placeholders
print(f"Predicted Genre: {genres[predicted_genre]}")
```


#### Output Generation

The final stage presents the results of the analysis in a usable format. This could be anything from a simple text output to a complex visualization:

```python
print(f"Tempo: {tempo} BPM")
```

### Design Considerations

When designing an MIR pipeline, several key considerations influence its performance and effectiveness.

#### Real-time vs Offline

* **Real-time**: Processing occurs as the audio is captured (e.g., live effects). Requires low latency.
* **Offline**: Processing happens after the audio is recorded (e.g., music analysis). Allows for more complex computations.

#### Accuracy vs Speed

* **Accuracy**:  Prioritizing correct results, often at the cost of processing time.
* **Speed**:  Favoring fast processing, potentially sacrificing some accuracy.

#### Memory Usage

Large audio files and complex features can consume significant memory.  Efficient data structures and algorithms are crucial for managing memory effectively.


### Error Handling Strategies

Robust error handling is paramount in any MIR pipeline. Here are some strategies:

* **Input Validation:** Ensure audio files are valid and in the correct format.
* **Exception Handling:** Use `try-except` blocks to catch potential errors during processing (e.g., `IOError`, `librosa.util.exceptions.ParameterError`).
* **Logging:**  Record errors and warnings for debugging and monitoring.

```python
import librosa
import logging

logging.basicConfig(level=logging.INFO)

def extract_features(filepath):
    try:
        audio, sr = librosa.load(filepath)
        # ... feature extraction code ...
        return features
    except (IOError, librosa.util.exceptions.ParameterError) as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None
```

### Practice

1. Build a simple pipeline that loads an audio file, calculates the MFCCs, and saves them to a file.
2. Experiment with different feature extraction techniques (e.g., spectral centroid, chroma features).
3. Implement error handling for invalid file paths and incorrect parameter values.
## Your First MIR Project

This section guides you through a basic Music Information Retrieval (MIR) project using Python.  Imagine you have a folder full of music files, and you want to automatically organize them by genre (rock, pop, jazz, etc.). This is a classic MIR task – genre classification – and a great starting point for understanding how MIR works in practice.  We'll build a simplified version of this classifier, focusing on the fundamental steps involved in any MIR project. This initial project will serve as a solid foundation for more complex tasks later in the book.

### Project Setup

Before diving into the code, it's crucial to set up your project correctly. A well-organized project saves you time and prevents headaches down the line. This process mirrors setting up any Python project, but with a focus on audio and MIR-specific libraries.

1. **Create a Virtual Environment:** This isolates your project's dependencies.  If you're familiar with virtual environments, feel free to skip ahead. If not, consider this similar to creating a sandbox for your code to play in without affecting other projects.

   ```bash
   python3 -m venv .venv  # Creates a virtual environment named '.venv'
   source .venv/bin/activate  # Activates the environment (Linux/macOS)
   .venv\Scripts\activate  # Activates the environment (Windows)
   ```

2. **Install Libraries:**  We’ll need `librosa` for audio analysis and `numpy` for numerical operations (covered in Chapter 2).

   ```bash
   pip install librosa numpy
   ```

3. **Organize Your Data:** Create a directory for your audio files. For this project, a small dataset is sufficient. You can find free music datasets online, or use a few of your own files. Organize them into subfolders representing genres (e.g., `rock/`, `pop/`, `jazz/`).

### Implementation Steps

Our simplified genre classifier will follow these steps, which are common in many MIR applications:

1. **Load Audio:** Read an audio file and convert it into a numerical representation that we can work with.

2. **Extract Features:** Calculate characteristics of the audio that distinguish different genres. For simplicity, we'll use the **spectral centroid**, a measure of the "brightness" of a sound (covered in Chapter 5).

3. **Classify:**  Use a simple rule-based approach to classify the genre based on the spectral centroid value.  We'll assume rock music has a lower centroid (darker) and pop music has a higher centroid (brighter).

```python
import librosa
import numpy as np

def classify_genre(filepath):
    """Classifies a music file as 'rock' or 'pop' based on spectral centroid."""
    try:
        y, sr = librosa.load(filepath)  # Load audio file
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()

        if centroid < 5000:  # Arbitrarily chosen threshold
            return "rock"
        else:
            return "pop"

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# Example usage
filepath = "music_files/rock/rock_song.mp3"
genre = classify_genre(filepath)
if genre:
    print(f"Genre: {genre}")


filepath = "music_files/pop/pop_song1.wav"      
genre = classify_genre(filepath)
if genre:
    print(f"Genre: {genre}")

```

### Testing and Validation

For a simple project like this, you can test by running the `classify_genre` function on a few audio files from each genre and checking if the output makes sense.  As you progress to more complex projects, formal evaluation using metrics like accuracy, precision, and recall becomes crucial. This will be explored in Chapter 10.

### Common Challenges

* **File Format Issues:**  `librosa` supports many formats, but you might encounter unsupported codecs.  Solutions: Use format conversion tools (e.g., `ffmpeg`) or try different libraries like `pydub`.
* **Corrupted Files:** Invalid audio files can cause errors.  Solution: Implement error handling (as shown in the code example) and consider using data validation tools.
* **Feature Selection:**  Choosing the right features significantly impacts classification accuracy. Experiment with different features (see Chapter 9).

### Best Practices

* **Modular Code:**  Break down tasks into reusable functions (as shown above).
* **Error Handling:**  Always include `try-except` blocks to catch potential errors.
* **Documentation:**  Write clear comments explaining your code.
* **Version Control:**  Use Git to track changes and collaborate effectively.

### Case Study: Building a Mood Classifier

Let's consider how you might apply these steps to classify music by mood (e.g., happy, sad).  You could extract features related to tempo and energy.  Faster tempo and high energy might indicate a happy mood, while slower tempo and low energy might suggest sadness.  This illustrates how different features can be used for different MIR tasks.

```python
import librosa
import numpy as np

def classify_mood(filepath):
    """Classifies music by mood (happy/sad) using tempo and energy."""
    y, sr = librosa.load(filepath)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0].mean() # Root Mean Square Energy

    if tempo > 120 and rms > 0.1: # Example threshold - needs tuning!
        return "happy"
    else:
        return "sad"


# Example usage: Note, needs error handling similar to classify_genre
filepath_happy = "music_files/happy/happy_song.mp3"
mood = classify_mood(filepath_happy)
print(f"Mood: {mood}")


filepath_sad = "music_files/sad/sad_song.wav"
mood = classify_mood(filepath_sad)
print(f"Mood: {mood}")
```

*Note: This is a simplified demonstration of mood classification. Actual mood classification is often far more complex.*


### Practice

1. Experiment with different threshold values in `classify_genre` and observe how it affects classification.
2. Try classifying a larger dataset of music files.
3. Explore other audio features from `librosa` and try incorporating them into your classifier.
