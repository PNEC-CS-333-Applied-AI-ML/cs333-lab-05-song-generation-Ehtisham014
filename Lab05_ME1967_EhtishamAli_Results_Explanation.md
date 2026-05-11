# Lab 05 — Music Generation with RNNs

**Course:** MIT Introduction to Deep Learning  
**Topic:** Sequence Modelling | Character-level LSTM | ABC Music Notation

---

## Overview

This lab implements a character-level Recurrent Neural Network (RNN) trained on a corpus of Irish folk songs represented in **ABC notation**. The trained model is then used to generate new, original music by predicting one character at a time.

---

## Objectives

- Understand character-level language modelling as a sequence classification task
- Build and train an LSTM-based RNN using TensorFlow/Keras
- Generate novel music in ABC notation and convert it to audio (`.wav`)
- Explore the effect of hyperparameters (epochs, batch size, learning rate, temperature) on generation quality

---

## Repository Contents

| File | Description |
|------|-------------|
| `Lab05_Music_Generation_ME1967_EhtishamAli.ipynb` | Main Jupyter notebook with all code, answers, and implementation |
| `Lab05_ME1967_EhtishamAli_OUTPUTS.zip` | Generated audio outputs and intermediate files |
| `Lab05_ME1967_EhtishamAli_Results_Explanation.md` | This file — explanation of the lab, methodology, and results |

### Contents of `OUTPUTS.zip`

| File | Description |
|------|-------------|
| `song.abc` | Generated song in raw ABC notation |
| `song.mid` | MIDI conversion of the generated song |
| `song.wav` | Full-length WAV of the generated song |
| `output_0.wav` | First valid extracted song snippet (short) |
| `output_2.wav` | Third extracted song snippet (longest — ~22 MB) |
| `output_3.wav` | Fourth extracted song snippet |
| `tmp.wav` | Intermediate render file |

---

## Methodology

### 1. Dataset
A collection of thousands of Irish folk songs in ABC notation was loaded using the `mitdeeplearning` library. Songs were joined into a single string and a character-level vocabulary (83 unique characters) was extracted.

### 2. Vectorization
Each character was mapped to a unique integer index via `char2idx`. The full song corpus was converted into a numeric array using this mapping, enabling the model to process it as a sequence prediction task.

### 3. Batch Generation
Training batches were constructed by randomly sampling starting indices from the vectorized corpus. Each input sequence of length `seq_length` is paired with a target sequence shifted one character to the right — the model learns to predict the next character at each step.

### 4. Model Architecture

```
Embedding(vocab_size=83, embedding_dim=256)
    ↓
LSTM(units=1024, return_sequences=True, stateful=True)
    ↓
Dense(vocab_size=83)   ← raw logits over vocabulary
```

The Embedding layer transforms integer indices into dense continuous vectors. The LSTM maintains a hidden state across time steps to capture sequential dependencies. The Dense layer outputs a distribution over all characters at each step.

### 5. Training

| Hyperparameter | Value |
|----------------|-------|
| Training iterations | 3000 |
| Batch size | 8 |
| Sequence length | 100 |
| Learning rate | 5e-3 |
| Embedding dim | 256 |
| RNN units | 1024 |
| Optimizer | Adam |
| Loss function | Sparse Categorical Crossentropy |

Model weights were checkpointed every 100 iterations.

### 6. Inference & Music Generation

The trained model was rebuilt with `batch_size=1` and the latest checkpoint weights were restored. Starting from the seed string `"X"` (the standard beginning of an ABC file), 1000 characters were generated autoregressively:

1. The seed is vectorized and fed to the model
2. The model outputs a distribution over the next character
3. The distribution is scaled by **temperature = 0.8** (lower → more conservative; higher → more random)
4. A character is sampled from the scaled distribution (not argmax, to avoid repetition loops)
5. The sampled character is appended to the generated sequence and fed back as the next input
6. Steps 2–5 repeat for 1000 characters

The generated ABC text was parsed for valid song snippets and each was synthesized to a WAV file using `abc2midi` + `timidity`.

---

## Results

### Generated Output
Multiple valid song snippets were extracted from the 1000-character generated sequence. The generated songs demonstrate the following characteristics of successful training:

- **Correct ABC structure** — proper headers (`X:`, `T:`, `M:`, `K:`), bar lines (`|`), and note groups
- **Plausible melodic phrasing** — note sequences that loosely follow the rhythmic patterns of Irish folk music
- **Variable length** — snippets ranged from short phrases (`output_0.wav`, ~2.8 MB) to longer compositions (`output_2.wav`, ~22.7 MB)

### Discussion of Key Concepts

**Effect of vocabulary size on learning complexity**  
The dataset contains 83 unique characters. Each character is a separate output class, so a larger vocabulary increases the size of the Embedding table and the final Dense layer's output dimension. This makes the classification problem harder — the model must distinguish between more possible next characters at each step — requiring more model capacity and training data for accurate predictions.

**Why untrained predictions are nonsensical**  
Before training, the model weights are randomly initialised (Glorot uniform for the LSTM recurrent kernel). The predictions are effectively random samples over the vocabulary, producing character sequences with no learned structure. Backpropagation through the sparse categorical cross-entropy loss progressively adjusts the weights so that the model assigns higher probability to the correct next character given its context.

**Effect of number of training epochs**  
- *Few epochs:* The model has not converged; output is mostly invalid or random ABC syntax with poor structure.
- *More epochs:* The model learns note patterns, bar structure, and valid ABC formatting, producing coherent and musically plausible songs.
- *Too many epochs:* Risk of overfitting — the model may reproduce training songs verbatim instead of generating novel compositions.

**Importance of the start string**  
The start string primes the LSTM's hidden state before generation begins. Using `"X"` is appropriate because all ABC files begin with `X:` (the song index field). An invalid or unusual start string can push the model into a region of state space it rarely visited during training, leading to early generation of invalid syntax that the autoregressive loop cannot recover from.

**Effect of dataset augmentation**  
Adding more or more varied folk songs (e.g., different regional styles, time signatures) gives the model a broader set of patterns to learn. This improves generalisation — the model becomes less likely to overfit to the specific Irish folk idiom — and increases variety in generated output.

---

## Dependencies

```
tensorflow >= 2.0
mitdeeplearning
numpy
scipy
tqdm
abcmidi
timidity
```

---

## How to Run

1. Open the notebook in **Google Colab** (GPU runtime recommended)
2. Run all cells in order
3. Training (~3000 iterations) will take a few minutes on GPU
4. Generated `.wav` files are saved automatically and linked in the notebook output
