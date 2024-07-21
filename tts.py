import gradio as gr
import librosa
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm
import pickle
import os
import gzip

n_mfcc = 90
context_length = 64
block_size = n_mfcc

# markov_chain = {
#     "hello": {
#         {
#             // mel spectrograms
#         }
#     }
# }
markov_chain = {}

# array level levenshtein distance, if the array has a subarray, it will calculate the levenshtein distance between the subarrays too
def levenshtein_distance(s1, s2, max_distance=None):
    if max_distance is None:
        max_distance = len(s1) + len(s2)
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    d = np.zeros((len(s1) + 1, len(s2) + 1))
    for i in range(len(s1) + 1):
        d[i][0] = i
    for i in range(len(s2) + 1):
        d[0][i] = i
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if np.array_equal(s1[i - 1], s2[j - 1]) else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if d[i][j] > max_distance:
                return max_distance
    return d[len(s1)][len(s2)]

def get_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=n_mfcc)
    return mfcc

def get_mfcc_from_file(file_path):
    audio, sr = librosa.load(file_path)
    return get_mfcc(audio)

# def syllable_count(word):
#     word = word.lower()
#     count = 0
#     vowels = "aeiouy"
#     if word[0] in vowels:
#         count += 1
#     for index in range(1, len(word)):
#         if word[index] in vowels and word[index - 1] not in vowels:
#             count += 1
#     if word.endswith("e") and not word.endswith("le"):
#         count -= 1
#     if count == 0:
#         count += 1
#     return count


# def split_into_syllables(word):
#     word = word.lower()
#     if len(word) < 6 or "'" in word:
#         return [word]
#     vowels = "aeiouy"
#     syllables = []
#     syllable = ""
#     if word[0] in vowels:
#         syllable += word[0]
#     for index in range(1, len(word)):
#         if word[index] in vowels and word[index - 1] not in vowels:
#             syllables.append(syllable)
#             syllable = ""
#         syllable += word[index]
#     if word.endswith("e") and not word.endswith("le"):
#         syllables.remove(syllables[-1])
#     syllables.append(syllable)
#     if (syllables[0] == ''):
#         syllables.remove(syllables[0])
#         syllables[0] = word[0] + syllables[0]
#     return syllables

# train tts model
def train_model(multiple_input_audios_path, epoch_count):
    global markov_chain
    # get all files in the directory
    audio_files = os.listdir(multiple_input_audios_path)
    # get all audio files
    audio_files = [f for f in audio_files if f.endswith(".wav")]
    # get all text files (transcriptions)
    text_files = [f.replace(".wav", ".txt") for f in audio_files]
    # get all
    for i in tqdm(range(len(audio_files))):
        audio_file = audio_files[i]
        text_file = text_files[i]
        # get mfcc
        mfcc = get_mfcc_from_file(os.path.join(multiple_input_audios_path, audio_file))
        if os.path.exists(os.path.join(multiple_input_audios_path, text_file)):
            with open(os.path.join(multiple_input_audios_path, text_file), "r") as f:
                text = f.read()
        else:
            text = audio_file.replace(".wav", "").split("-")[0].lower()
        # split text into words
        words = text.split(" ")
        # iterate over words
        for word in words:
            if word not in markov_chain:
                markov_chain[word] = []
            markov_chain[word].append(mfcc)
    # save model
    with gzip.open("tts_model.pkl", "wb") as f:
        pickle.dump(markov_chain, f)
    return "Model Trained"

def generate_sequence_from_gradio(input_prompt, length, audio_seed, random_seed):
    global markov_chain
    if random_seed is not None and random_seed != 0:
        random.seed(random_seed)
    # load model
    with gzip.open("tts_model.pkl", "rb") as f:
        markov_chain = pickle.load(f)
    # get mfcc from audio seed
    if audio_seed is not None:
        audio_seed, sr = librosa.load(audio_seed)
        mfcc_seed = get_mfcc(audio_seed)
    else:
        mfcc_seed = None
    input_prompt = input_prompt.lower().replace(".", "").replace(",", "").replace("-", "_").replace("/", "_").replace("?", "_")
    # split input prompt into words
    words = input_prompt.split(" ")
    # get the last word
    last_word = words[-1]
    # get the mfccs
    mfccs = markov_chain[last_word]
    # get random mfcc
    mfcc = random.choice(mfccs)
    # get random mfcc from the seed
    sequence = []
    sequence.append(random.choice(mfccs))
    if mfcc_seed is not None:
        sequence.append(mfcc_seed)
    
    for i in range(len(words)):
        next_word = words[i % len(words)]
        # get the last mfcc
        last_mfcc = sequence[-1]
        # get the next mfccs
        next_mfccs = markov_chain[next_word]
        # get distances
        distances = [levenshtein_distance(last_mfcc, next_mfcc) for next_mfcc in next_mfccs]
        # get the closest mfcc
        next_mfcc = next_mfccs[np.argmin(distances)]
        # add the mfcc to the sequence
        sequence.append(next_mfcc)
    sequence = sequence[1:]
    # if length is not 0, generate extra sequence
    if length != 0:
        for i in range(length):
            # get the last mfcc
            last_mfcc = sequence[-1]
            # get current index
            current_index = list(markov_chain.keys()).index(words[i % len(words)])
            # get the next index
            best_index = None
            best_distance = None
            tries_left = 25
            for i in range(tries_left):
                if current_index >= len(list(markov_chain.keys())):
                    current_index = 0
                ind = random.randint(current_index, len(list(markov_chain.keys())) - 1)
                next_mfcc = markov_chain[list(markov_chain.keys())[ind]][0]
                distance = levenshtein_distance(last_mfcc, next_mfcc)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = ind
                if best_distance == 0:
                    break
            current_index = best_index
            # add the mfcc to the sequence
            sequence.append(markov_chain[list(markov_chain.keys())[best_index]][0])
            
    # generate audio
    audio = np.zeros(0)
    for mfcc in sequence:
        audio_block = librosa.feature.inverse.mfcc_to_audio(mfcc)
        audio = np.concatenate((audio, audio_block))
    # save audio
    sf.write("output.wav", audio, 22050)
    return "output.wav"


with gr.Blocks() as block:
    with gr.Tab("Generate Sequence"):
        input_prompt = gr.Textbox(label="Input Prompt", type="text")
        output_audio = gr.Audio(label="Output Audio")
        length = gr.Slider(0, 100, 0, 0, label="Length of Extra Sequence")
        btn = gr.Button("Generate Sequence")
        audio_seed = gr.Audio(label="Continue from Audio Seed", type="filepath")
        actual_random_seed = gr.Number(label="Random Seed")
        btn.click(fn=generate_sequence_from_gradio, inputs=[input_prompt, length, audio_seed, actual_random_seed], outputs=[output_audio])
    with gr.Tab("Train Model"):
        multiple_input_audios_path = gr.Textbox(label="Dataset Path", type="text")
        btn = gr.Button("Train Model")
        epoch_count = gr.Slider(1, 500, 5, 1, label="Epoch Count")
        btn.click(fn=train_model, inputs=[multiple_input_audios_path, epoch_count], outputs=[gr.Textbox(label="Model Trained")])

block.launch(server_name="0.0.0.0", server_port=7878)