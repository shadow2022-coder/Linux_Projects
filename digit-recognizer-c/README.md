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

## What this project includes

- A handwritten digit CNN written in plain C
- End-to-end MNIST training and evaluation in the C executable
- Saved binary weights for reuse without retraining every time
- A lightweight HTTP server written in C
- A browser drawing UI for interactive digit prediction

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

## Verification

This version was tested end to end on July 18, 2026 with the following flow:

```sh
make
./digitrecg train --data-dir data --epochs 1 --batch-size 64 --learning-rate 0.01 --weights model_weights.bin --sample-index 7
./digitrecg predict --data-dir data --weights model_weights.bin --sample-index 7
./digitrecg serve --weights model_weights.bin --port 8081
curl -s http://127.0.0.1:8081/health
curl -s -X POST http://127.0.0.1:8081/api/predict -H 'Content-Type: application/json' --data-binary @web_request.txt
```

Observed result from the CNN training smoke test:

- Test accuracy after 1 epoch: `89.60%`
- The saved model correctly predicted MNIST test sample `7` as digit `9`
- The HTTP server returned a healthy response and the prediction API returned the same digit through the web path

## Notes

- The `.mat` files are raw `float32` dumps, not MATLAB container files.
- The server is written in C and serves the web UI plus a `/api/predict` inference endpoint.
- Trained weights are stored in `model_weights.bin` and are not committed.
