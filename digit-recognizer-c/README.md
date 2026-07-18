# CNN Digit Recognizer in C

This project trains a real convolutional neural network in plain C, saves the learned weights, and serves a small browser UI from a C HTTP server so you can draw digits and test the model interactively.

## Architecture

- `Conv 3x3, 8 filters`
- `ReLU`
- `2x2 max pooling`
- `Dense 32`
- `ReLU`
- `Dense 10`
- `Softmax`

## Requirements

- `cc` or `clang`
- Python 3

## Quick start

```sh
make prepare-data
make
./digitrecg train --data-dir data --epochs 2 --batch-size 64 --learning-rate 0.01 --weights model_weights.bin
./digitrecg serve --weights model_weights.bin --port 8080
```

Then open `http://127.0.0.1:8080`.

## Useful commands

```sh
make prepare-data
make
make train
make predict
make serve
```

Manual examples:

```sh
./digitrecg train --data-dir data --epochs 3 --batch-size 64 --learning-rate 0.01 --weights model_weights.bin
./digitrecg predict --data-dir data --weights model_weights.bin --sample-index 9
./digitrecg serve --weights model_weights.bin --port 8080
```

## Notes

- The `.mat` files are raw `float32` dumps, not MATLAB container files.
- The server is written in C and serves the web UI plus a `/api/predict` inference endpoint.
- Trained weights are stored in `model_weights.bin` and are not committed.
