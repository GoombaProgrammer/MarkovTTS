# Novixx's MarkovTTS, a Markov Chain based Text-to-Speech synthesizer

This is a simple text-to-speech synthesizer that uses a Markov Chain to generate speech. It written in python
and uses gradio for the web interface. Use it to train or use a Markov Chain model to generate speech.

## Limitations
- The model is not very good at generating speech when the model has multiple voices, make sure your data has 1 voice.
- The model size increases exponentially making it hard to train on large datasets.

## Usage
1. Install the requirements:
```bash
pip install gradio numpy librosa tqdm gzip soundfile
```
2. Run the script:
```bash
python tts.py
```

## Dataset format
The dataset should be a directory containing WAV files, and optionally TXT files with the same name as the WAV file.
The TXT file should contain the transcript of the WAV file, if a file has no corresponding TXT file, the filename will be used as the transcript.

## Training
To train the model, run the script, navigate to the web interface, and click the "Train" tab. Select the dataset directory and click "Train Model".

# License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

# Acknowledgements
This project was made possible by the following libraries:
- [Gradio](https://gradio.app/) for the web interface.
- [Librosa](https://librosa.org/) for audio processing.

## Contributors
- [GoombaProgrammer](https://www.github.com/GoombaProgrammer)

(Add your name here in your first PR, or not, if you don't want to be listed)

