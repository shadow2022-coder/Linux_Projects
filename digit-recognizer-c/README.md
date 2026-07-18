# Digit Recognizer in C

This project trains a small fully-connected neural network in plain C to classify handwritten MNIST digits.

## Requirements

- `cc` or `clang`
- Python 3

## Quick start

```sh
make prepare-data
make
make run
```

## Manual run

```sh
./digitrecg --data-dir data
```

Optional flags:

```sh
./digitrecg --data-dir data --epochs 5 --batch-size 64 --learning-rate 0.02 --sample-index 7
```

## Notes

- The `.mat` files used here are raw `float32` dumps, not MATLAB container files.
- If the dataset files are missing, the C program prints the exact generation command to run.
- The default run trains on the full MNIST dataset, so it can take a while.
