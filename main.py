import outetts
import sounddevice as sd
import numpy as np
import tempfile
import wave


# Configure the model
model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",  
)

interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)
interface.print_default_speakers()
speaker = interface.load_default_speaker(name="female_2")

# Function to generate and play audio from text
def generate_and_play_audio(text):
    # Generate the TTS output for the provided text
    output = interface.generate(
        text=text,
        temperature=0.5,
        repetition_penalty=1.1,
        max_length=4096,
        speaker=speaker,
    )

    # Save to a temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output.save(temp_file.name)

    # Stream audio playback
    play_audio(temp_file.name)

# Function to play audio from a given WAV file path
def play_audio(file_path):
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        dtype = 'int16' if wf.getsampwidth() == 2 else 'int32'

        def callback(outdata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            data = wf.readframes(frames)
            if len(data) == 0:
                raise sd.CallbackStop()
            outdata[:len(data)] = np.frombuffer(data, dtype=dtype).reshape(-1, num_channels)

        with sd.OutputStream(samplerate=sample_rate, channels=num_channels, callback=callback, dtype=dtype):
            print("Playing audio...")
            sd.sleep(int(wf.getnframes() / sample_rate * 1000))

# Main function to process incoming text sentences
def main():
    sentences = [
        "Speech synthesis is the artificial production of human speech.",
        "A computer system used for this purpose is called a speech synthesizer.",
        "It can be implemented in software or hardware products."
    ]

    for sentence in sentences:
        print(f"Generating speech for: {sentence}")
        generate_and_play_audio(sentence)

# Entry point of the script
if __name__ == "__main__":
    main()
