#define _CRT_SECURE_NO_WARNINGS

#include "base.h"
#include "prng.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

enum {
    MNIST_IMAGE_SIDE = 28,
    MNIST_INPUT_SIZE = 28 * 28,
    MNIST_OUTPUT_SIZE = 10,
    MNIST_TRAIN_IMAGES = 60000,
    MNIST_TEST_IMAGES = 10000,

    CNN_FILTERS = 8,
    CNN_KERNEL = 3,
    CNN_CONV_SIDE = MNIST_IMAGE_SIDE - CNN_KERNEL + 1,
    CNN_POOL_STRIDE = 2,
    CNN_POOL_SIDE = CNN_CONV_SIDE / CNN_POOL_STRIDE,
    CNN_FEATURES = CNN_FILTERS * CNN_POOL_SIDE * CNN_POOL_SIDE,
    CNN_HIDDEN_UNITS = 32,

    HTTP_BUFFER_SIZE = 1 << 20,
};

typedef enum {
    APP_MODE_TRAIN = 0,
    APP_MODE_PREDICT,
    APP_MODE_SERVE,
} app_mode;

typedef struct {
    app_mode mode;
    const char* data_dir;
    const char* weights_path;
    const char* web_dir;
    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
    u32 sample_index;
    u16 port;
} app_config;

typedef enum {
    PARSE_ARGS_OK = 0,
    PARSE_ARGS_HELP,
    PARSE_ARGS_ERROR,
} parse_args_result;

typedef struct {
    f32* images;
    u8* labels;
    u32 count;
} mnist_split;

typedef struct {
    f32* conv_w;
    f32* conv_b;
    f32* fc1_w;
    f32* fc1_b;
    f32* fc2_w;
    f32* fc2_b;
} cnn_model;

typedef struct {
    f32* conv_w;
    f32* conv_b;
    f32* fc1_w;
    f32* fc1_b;
    f32* fc2_w;
    f32* fc2_b;
} cnn_grads;

typedef struct {
    f32* conv_pre;
    f32* conv_act;
    f32* pool;
    i32* pool_switch;
    f32* hidden_pre;
    f32* hidden_act;
    f32* logits;
    f32* probs;
} cnn_cache;

typedef struct {
    char magic[8];
    u32 version;
    u32 filters;
    u32 kernel;
    u32 hidden_units;
} weights_header;

static volatile sig_atomic_t g_keep_running = 1;

static inline u32 conv_index(u32 filter, u32 y, u32 x) {
    return filter * CNN_CONV_SIDE * CNN_CONV_SIDE + y * CNN_CONV_SIDE + x;
}

static inline u32 pool_index(u32 filter, u32 y, u32 x) {
    return filter * CNN_POOL_SIDE * CNN_POOL_SIDE + y * CNN_POOL_SIDE + x;
}

static void log_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

static void signal_handler(i32 signum) {
    (void)signum;
    g_keep_running = 0;
}

static void print_usage(const char* argv0) {
    printf(
        "Usage:\n"
        "  %s train [options]\n"
        "  %s predict [options]\n"
        "  %s serve [options]\n\n"
        "Common options:\n"
        "  --data-dir <path>        Directory containing MNIST .mat files (default: data)\n"
        "  --weights <path>         Weights file path (default: model_weights.bin)\n"
        "  --sample-index <index>   Test sample to preview/predict (default: 7)\n\n"
        "Training options:\n"
        "  --epochs <count>         Training epochs (default: 2)\n"
        "  --batch-size <count>     Mini-batch size (default: 64)\n"
        "  --learning-rate <value>  SGD learning rate (default: 0.01)\n\n"
        "Web server options:\n"
        "  --web-dir <path>         Directory containing web assets (default: web)\n"
        "  --port <number>          HTTP port for serve mode (default: 8080)\n\n"
        "Examples:\n"
        "  %s train --data-dir data --epochs 3 --weights model_weights.bin\n"
        "  %s predict --data-dir data --weights model_weights.bin --sample-index 9\n"
        "  %s serve --weights model_weights.bin --port 8080\n",
        argv0, argv0, argv0, argv0, argv0, argv0
    );
}

static b32 parse_u32_arg(const char* text, u32* out_value) {
    char* end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (text[0] == '\0' || end == text || *end != '\0' || value > UINT32_MAX) {
        return false;
    }
    *out_value = (u32)value;
    return true;
}

static b32 parse_u16_arg(const char* text, u16* out_value) {
    u32 value = 0;
    if (!parse_u32_arg(text, &value) || value > UINT16_MAX) {
        return false;
    }
    *out_value = (u16)value;
    return true;
}

static b32 parse_f32_arg(const char* text, f32* out_value) {
    char* end = NULL;
    errno = 0;
    float value = strtof(text, &end);
    if (text[0] == '\0' || end == text || *end != '\0' || errno != 0) {
        return false;
    }
    *out_value = value;
    return true;
}

static parse_args_result parse_args(i32 argc, char** argv, app_config* config) {
    i32 arg_index = 1;

    if (arg_index < argc && argv[arg_index][0] != '-') {
        if (strcmp(argv[arg_index], "train") == 0) {
            config->mode = APP_MODE_TRAIN;
        } else if (strcmp(argv[arg_index], "predict") == 0) {
            config->mode = APP_MODE_PREDICT;
        } else if (strcmp(argv[arg_index], "serve") == 0) {
            config->mode = APP_MODE_SERVE;
        } else {
            fprintf(stderr, "Unknown mode: %s\n", argv[arg_index]);
            print_usage(argv[0]);
            return PARSE_ARGS_ERROR;
        }
        arg_index++;
    }

    for (i32 i = arg_index; i < argc; i++) {
        const char* arg = argv[i];

        if (strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return PARSE_ARGS_HELP;
        } else if (strcmp(arg, "--data-dir") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--data-dir requires a value.\n");
                return PARSE_ARGS_ERROR;
            }
            config->data_dir = argv[++i];
        } else if (strcmp(arg, "--weights") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--weights requires a value.\n");
                return PARSE_ARGS_ERROR;
            }
            config->weights_path = argv[++i];
        } else if (strcmp(arg, "--web-dir") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--web-dir requires a value.\n");
                return PARSE_ARGS_ERROR;
            }
            config->web_dir = argv[++i];
        } else if (strcmp(arg, "--epochs") == 0) {
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &config->epochs) || config->epochs == 0) {
                fprintf(stderr, "--epochs requires a positive integer.\n");
                return PARSE_ARGS_ERROR;
            }
        } else if (strcmp(arg, "--batch-size") == 0) {
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &config->batch_size) || config->batch_size == 0) {
                fprintf(stderr, "--batch-size requires a positive integer.\n");
                return PARSE_ARGS_ERROR;
            }
        } else if (strcmp(arg, "--learning-rate") == 0) {
            if (i + 1 >= argc || !parse_f32_arg(argv[++i], &config->learning_rate) || config->learning_rate <= 0.0f) {
                fprintf(stderr, "--learning-rate requires a positive number.\n");
                return PARSE_ARGS_ERROR;
            }
        } else if (strcmp(arg, "--sample-index") == 0) {
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &config->sample_index)) {
                fprintf(stderr, "--sample-index requires a non-negative integer.\n");
                return PARSE_ARGS_ERROR;
            }
        } else if (strcmp(arg, "--port") == 0) {
            if (i + 1 >= argc || !parse_u16_arg(argv[++i], &config->port) || config->port == 0) {
                fprintf(stderr, "--port requires a valid port number.\n");
                return PARSE_ARGS_ERROR;
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg);
            print_usage(argv[0]);
            return PARSE_ARGS_ERROR;
        }
    }

    return PARSE_ARGS_OK;
}

static b32 build_data_path(char* out, u64 out_size, const char* data_dir, const char* filename) {
    i32 written = snprintf(out, (size_t)out_size, "%s/%s", data_dir, filename);
    return written > 0 && (u64)written < out_size;
}

static void draw_mnist_digit(const f32* data) {
    for (u32 y = 0; y < MNIST_IMAGE_SIDE; y++) {
        for (u32 x = 0; x < MNIST_IMAGE_SIDE; x++) {
            f32 value = data[x + y * MNIST_IMAGE_SIDE];
            u32 color = 232 + (u32)(value * 23.0f);
            printf("\x1b[48;5;%dm  ", color);
        }
        printf("\n");
    }
    printf("\x1b[0m");
}

static f32* load_f32_file(const char* filename, u64 expected_count) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        log_error("Failed to open %s\n", filename);
        return NULL;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        log_error("Failed to seek %s\n", filename);
        return NULL;
    }

    long file_size = ftell(file);
    if (file_size < 0) {
        fclose(file);
        log_error("Failed to determine the size of %s\n", filename);
        return NULL;
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        log_error("Failed to rewind %s\n", filename);
        return NULL;
    }

    u64 expected_size = expected_count * sizeof(f32);
    if ((u64)file_size < expected_size) {
        fclose(file);
        log_error(
            "%s is too small. Expected at least %llu bytes, got %ld.\n",
            filename,
            (unsigned long long)expected_size,
            file_size
        );
        return NULL;
    }

    f32* data = (f32*)malloc((size_t)expected_size);
    if (data == NULL) {
        fclose(file);
        log_error("Out of memory while loading %s\n", filename);
        return NULL;
    }

    size_t read_bytes = fread(data, 1, (size_t)expected_size, file);
    fclose(file);

    if (read_bytes != (size_t)expected_size) {
        free(data);
        log_error("Failed to read %s completely.\n", filename);
        return NULL;
    }

    return data;
}

static u8* load_label_file(const char* filename, u64 expected_count) {
    f32* labels_raw = load_f32_file(filename, expected_count);
    if (labels_raw == NULL) {
        return NULL;
    }

    u8* labels = (u8*)malloc((size_t)expected_count);
    if (labels == NULL) {
        free(labels_raw);
        return NULL;
    }

    for (u64 i = 0; i < expected_count; i++) {
        i32 label = (i32)(labels_raw[i] + 0.5f);
        if (label < 0 || label >= MNIST_OUTPUT_SIZE) {
            log_error("Invalid label %d in %s at index %llu\n", label, filename, (unsigned long long)i);
            free(labels_raw);
            free(labels);
            return NULL;
        }
        labels[i] = (u8)label;
    }

    free(labels_raw);
    return labels;
}

static b32 load_mnist_split(
    const char* data_dir,
    const char* images_name,
    const char* labels_name,
    u32 count,
    mnist_split* out_split
) {
    char images_path[256] = { 0 };
    char labels_path[256] = { 0 };

    if (
        !build_data_path(images_path, sizeof(images_path), data_dir, images_name) ||
        !build_data_path(labels_path, sizeof(labels_path), data_dir, labels_name)
    ) {
        fprintf(stderr, "The dataset path is too long.\n");
        return false;
    }

    out_split->images = load_f32_file(images_path, (u64)count * MNIST_INPUT_SIZE);
    out_split->labels = load_label_file(labels_path, count);
    out_split->count = count;

    if (out_split->images == NULL || out_split->labels == NULL) {
        free(out_split->images);
        free(out_split->labels);
        out_split->images = NULL;
        out_split->labels = NULL;
        fprintf(
            stderr,
            "MNIST data files are missing or invalid.\n"
            "Run `python3 mnist.py --output-dir %s` first.\n",
            data_dir
        );
        return false;
    }

    return true;
}

static void free_mnist_split(mnist_split* split) {
    free(split->images);
    free(split->labels);
    memset(split, 0, sizeof(*split));
}

static b32 alloc_model(cnn_model* model) {
    memset(model, 0, sizeof(*model));

    model->conv_w = (f32*)calloc(CNN_FILTERS * CNN_KERNEL * CNN_KERNEL, sizeof(f32));
    model->conv_b = (f32*)calloc(CNN_FILTERS, sizeof(f32));
    model->fc1_w = (f32*)calloc(CNN_HIDDEN_UNITS * CNN_FEATURES, sizeof(f32));
    model->fc1_b = (f32*)calloc(CNN_HIDDEN_UNITS, sizeof(f32));
    model->fc2_w = (f32*)calloc(MNIST_OUTPUT_SIZE * CNN_HIDDEN_UNITS, sizeof(f32));
    model->fc2_b = (f32*)calloc(MNIST_OUTPUT_SIZE, sizeof(f32));

    return model->conv_w != NULL &&
        model->conv_b != NULL &&
        model->fc1_w != NULL &&
        model->fc1_b != NULL &&
        model->fc2_w != NULL &&
        model->fc2_b != NULL;
}

static b32 alloc_grads(cnn_grads* grads) {
    return alloc_model((cnn_model*)grads);
}

static void free_model(cnn_model* model) {
    free(model->conv_w);
    free(model->conv_b);
    free(model->fc1_w);
    free(model->fc1_b);
    free(model->fc2_w);
    free(model->fc2_b);
    memset(model, 0, sizeof(*model));
}

static void zero_grads(cnn_grads* grads) {
    memset(grads->conv_w, 0, sizeof(f32) * CNN_FILTERS * CNN_KERNEL * CNN_KERNEL);
    memset(grads->conv_b, 0, sizeof(f32) * CNN_FILTERS);
    memset(grads->fc1_w, 0, sizeof(f32) * CNN_HIDDEN_UNITS * CNN_FEATURES);
    memset(grads->fc1_b, 0, sizeof(f32) * CNN_HIDDEN_UNITS);
    memset(grads->fc2_w, 0, sizeof(f32) * MNIST_OUTPUT_SIZE * CNN_HIDDEN_UNITS);
    memset(grads->fc2_b, 0, sizeof(f32) * MNIST_OUTPUT_SIZE);
}

static b32 alloc_cache(cnn_cache* cache) {
    memset(cache, 0, sizeof(*cache));
    cache->conv_pre = (f32*)calloc(CNN_FILTERS * CNN_CONV_SIDE * CNN_CONV_SIDE, sizeof(f32));
    cache->conv_act = (f32*)calloc(CNN_FILTERS * CNN_CONV_SIDE * CNN_CONV_SIDE, sizeof(f32));
    cache->pool = (f32*)calloc(CNN_FEATURES, sizeof(f32));
    cache->pool_switch = (i32*)calloc(CNN_FEATURES, sizeof(i32));
    cache->hidden_pre = (f32*)calloc(CNN_HIDDEN_UNITS, sizeof(f32));
    cache->hidden_act = (f32*)calloc(CNN_HIDDEN_UNITS, sizeof(f32));
    cache->logits = (f32*)calloc(MNIST_OUTPUT_SIZE, sizeof(f32));
    cache->probs = (f32*)calloc(MNIST_OUTPUT_SIZE, sizeof(f32));

    return cache->conv_pre != NULL &&
        cache->conv_act != NULL &&
        cache->pool != NULL &&
        cache->pool_switch != NULL &&
        cache->hidden_pre != NULL &&
        cache->hidden_act != NULL &&
        cache->logits != NULL &&
        cache->probs != NULL;
}

static void free_cache(cnn_cache* cache) {
    free(cache->conv_pre);
    free(cache->conv_act);
    free(cache->pool);
    free(cache->pool_switch);
    free(cache->hidden_pre);
    free(cache->hidden_act);
    free(cache->logits);
    free(cache->probs);
    memset(cache, 0, sizeof(*cache));
}

static f32 rand_range(f32 lower, f32 upper) {
    return prng_randf() * (upper - lower) + lower;
}

static void init_model(cnn_model* model) {
    f32 conv_bound = sqrtf(6.0f / (CNN_KERNEL * CNN_KERNEL + CNN_FILTERS * CNN_KERNEL * CNN_KERNEL));
    f32 fc1_bound = sqrtf(6.0f / (CNN_FEATURES + CNN_HIDDEN_UNITS));
    f32 fc2_bound = sqrtf(6.0f / (CNN_HIDDEN_UNITS + MNIST_OUTPUT_SIZE));

    for (u32 i = 0; i < CNN_FILTERS * CNN_KERNEL * CNN_KERNEL; i++) {
        model->conv_w[i] = rand_range(-conv_bound, conv_bound);
    }

    for (u32 i = 0; i < CNN_HIDDEN_UNITS * CNN_FEATURES; i++) {
        model->fc1_w[i] = rand_range(-fc1_bound, fc1_bound);
    }

    for (u32 i = 0; i < MNIST_OUTPUT_SIZE * CNN_HIDDEN_UNITS; i++) {
        model->fc2_w[i] = rand_range(-fc2_bound, fc2_bound);
    }
}

static void cnn_forward(const cnn_model* model, const f32* image, cnn_cache* cache) {
    for (u32 filter = 0; filter < CNN_FILTERS; filter++) {
        for (u32 y = 0; y < CNN_CONV_SIDE; y++) {
            for (u32 x = 0; x < CNN_CONV_SIDE; x++) {
                f32 sum = model->conv_b[filter];
                for (u32 ky = 0; ky < CNN_KERNEL; ky++) {
                    for (u32 kx = 0; kx < CNN_KERNEL; kx++) {
                        u32 image_index = (y + ky) * MNIST_IMAGE_SIDE + (x + kx);
                        u32 weight_index = filter * CNN_KERNEL * CNN_KERNEL + ky * CNN_KERNEL + kx;
                        sum += model->conv_w[weight_index] * image[image_index];
                    }
                }

                u32 index = conv_index(filter, y, x);
                cache->conv_pre[index] = sum;
                cache->conv_act[index] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }

    for (u32 filter = 0; filter < CNN_FILTERS; filter++) {
        for (u32 y = 0; y < CNN_POOL_SIDE; y++) {
            for (u32 x = 0; x < CNN_POOL_SIDE; x++) {
                f32 best = -FLT_MAX;
                i32 best_index = -1;

                for (u32 py = 0; py < CNN_POOL_STRIDE; py++) {
                    for (u32 px = 0; px < CNN_POOL_STRIDE; px++) {
                        u32 source_y = y * CNN_POOL_STRIDE + py;
                        u32 source_x = x * CNN_POOL_STRIDE + px;
                        u32 source_index = conv_index(filter, source_y, source_x);
                        f32 candidate = cache->conv_act[source_index];
                        if (candidate > best) {
                            best = candidate;
                            best_index = (i32)source_index;
                        }
                    }
                }

                u32 index = pool_index(filter, y, x);
                cache->pool[index] = best;
                cache->pool_switch[index] = best_index;
            }
        }
    }

    for (u32 hidden = 0; hidden < CNN_HIDDEN_UNITS; hidden++) {
        f32 sum = model->fc1_b[hidden];
        u32 weight_base = hidden * CNN_FEATURES;
        for (u32 feature = 0; feature < CNN_FEATURES; feature++) {
            sum += model->fc1_w[weight_base + feature] * cache->pool[feature];
        }
        cache->hidden_pre[hidden] = sum;
        cache->hidden_act[hidden] = sum > 0.0f ? sum : 0.0f;
    }

    f32 max_logit = -FLT_MAX;
    for (u32 output = 0; output < MNIST_OUTPUT_SIZE; output++) {
        f32 sum = model->fc2_b[output];
        u32 weight_base = output * CNN_HIDDEN_UNITS;
        for (u32 hidden = 0; hidden < CNN_HIDDEN_UNITS; hidden++) {
            sum += model->fc2_w[weight_base + hidden] * cache->hidden_act[hidden];
        }
        cache->logits[output] = sum;
        if (sum > max_logit) {
            max_logit = sum;
        }
    }

    f32 softmax_sum = 0.0f;
    for (u32 output = 0; output < MNIST_OUTPUT_SIZE; output++) {
        cache->probs[output] = expf(cache->logits[output] - max_logit);
        softmax_sum += cache->probs[output];
    }

    for (u32 output = 0; output < MNIST_OUTPUT_SIZE; output++) {
        cache->probs[output] /= softmax_sum;
    }
}

static f32 cnn_backward(
    const cnn_model* model,
    cnn_grads* grads,
    const f32* image,
    u8 label,
    const cnn_cache* cache
) {
    f32 dlogits[MNIST_OUTPUT_SIZE] = { 0 };
    f32 dhidden[CNN_HIDDEN_UNITS] = { 0 };
    f32 dpool[CNN_FEATURES] = { 0 };
    f32 dconv[CNN_FILTERS * CNN_CONV_SIDE * CNN_CONV_SIDE] = { 0 };

    for (u32 output = 0; output < MNIST_OUTPUT_SIZE; output++) {
        dlogits[output] = cache->probs[output];
    }
    dlogits[label] -= 1.0f;

    f32 loss = -logf(cache->probs[label] > 1e-7f ? cache->probs[label] : 1e-7f);

    for (u32 output = 0; output < MNIST_OUTPUT_SIZE; output++) {
        grads->fc2_b[output] += dlogits[output];
        u32 weight_base = output * CNN_HIDDEN_UNITS;
        for (u32 hidden = 0; hidden < CNN_HIDDEN_UNITS; hidden++) {
            grads->fc2_w[weight_base + hidden] += dlogits[output] * cache->hidden_act[hidden];
            dhidden[hidden] += model->fc2_w[weight_base + hidden] * dlogits[output];
        }
    }

    for (u32 hidden = 0; hidden < CNN_HIDDEN_UNITS; hidden++) {
        if (cache->hidden_pre[hidden] <= 0.0f) {
            dhidden[hidden] = 0.0f;
        }
        grads->fc1_b[hidden] += dhidden[hidden];

        u32 weight_base = hidden * CNN_FEATURES;
        for (u32 feature = 0; feature < CNN_FEATURES; feature++) {
            grads->fc1_w[weight_base + feature] += dhidden[hidden] * cache->pool[feature];
            dpool[feature] += model->fc1_w[weight_base + feature] * dhidden[hidden];
        }
    }

    for (u32 feature = 0; feature < CNN_FEATURES; feature++) {
        i32 source_index = cache->pool_switch[feature];
        if (source_index >= 0) {
            dconv[source_index] += dpool[feature];
        }
    }

    for (u32 filter = 0; filter < CNN_FILTERS; filter++) {
        for (u32 y = 0; y < CNN_CONV_SIDE; y++) {
            for (u32 x = 0; x < CNN_CONV_SIDE; x++) {
                u32 index = conv_index(filter, y, x);
                if (cache->conv_pre[index] <= 0.0f) {
                    dconv[index] = 0.0f;
                }
                grads->conv_b[filter] += dconv[index];

                for (u32 ky = 0; ky < CNN_KERNEL; ky++) {
                    for (u32 kx = 0; kx < CNN_KERNEL; kx++) {
                        u32 image_index = (y + ky) * MNIST_IMAGE_SIDE + (x + kx);
                        u32 weight_index = filter * CNN_KERNEL * CNN_KERNEL + ky * CNN_KERNEL + kx;
                        grads->conv_w[weight_index] += dconv[index] * image[image_index];
                    }
                }
            }
        }
    }

    return loss;
}

static void apply_grads(cnn_model* model, const cnn_grads* grads, f32 scale) {
    for (u32 i = 0; i < CNN_FILTERS * CNN_KERNEL * CNN_KERNEL; i++) {
        model->conv_w[i] -= grads->conv_w[i] * scale;
    }
    for (u32 i = 0; i < CNN_FILTERS; i++) {
        model->conv_b[i] -= grads->conv_b[i] * scale;
    }
    for (u32 i = 0; i < CNN_HIDDEN_UNITS * CNN_FEATURES; i++) {
        model->fc1_w[i] -= grads->fc1_w[i] * scale;
    }
    for (u32 i = 0; i < CNN_HIDDEN_UNITS; i++) {
        model->fc1_b[i] -= grads->fc1_b[i] * scale;
    }
    for (u32 i = 0; i < MNIST_OUTPUT_SIZE * CNN_HIDDEN_UNITS; i++) {
        model->fc2_w[i] -= grads->fc2_w[i] * scale;
    }
    for (u32 i = 0; i < MNIST_OUTPUT_SIZE; i++) {
        model->fc2_b[i] -= grads->fc2_b[i] * scale;
    }
}

static u32 argmax10(const f32* probs) {
    u32 best_index = 0;
    for (u32 i = 1; i < MNIST_OUTPUT_SIZE; i++) {
        if (probs[i] > probs[best_index]) {
            best_index = i;
        }
    }
    return best_index;
}

static void shuffle_indices(u32* indices, u32 count) {
    for (u32 i = count - 1; i > 0; i--) {
        u32 j = prng_rand() % (i + 1);
        u32 temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

static void evaluate_model(
    const cnn_model* model,
    const mnist_split* split,
    cnn_cache* cache,
    u32* out_correct,
    f32* out_avg_loss
) {
    u32 correct = 0;
    f32 total_loss = 0.0f;

    for (u32 i = 0; i < split->count; i++) {
        const f32* image = split->images + (u64)i * MNIST_INPUT_SIZE;
        u8 label = split->labels[i];

        cnn_forward(model, image, cache);
        if (argmax10(cache->probs) == label) {
            correct++;
        }

        total_loss += -logf(cache->probs[label] > 1e-7f ? cache->probs[label] : 1e-7f);
    }

    *out_correct = correct;
    *out_avg_loss = total_loss / split->count;
}

static b32 save_weights(const cnn_model* model, const char* path) {
    FILE* file = fopen(path, "wb");
    if (file == NULL) {
        log_error("Failed to open %s for writing.\n", path);
        return false;
    }

    weights_header header = {
        .magic = { 'D', 'R', 'C', 'N', 'N', '0', '1', '\0' },
        .version = 1,
        .filters = CNN_FILTERS,
        .kernel = CNN_KERNEL,
        .hidden_units = CNN_HIDDEN_UNITS,
    };

    fwrite(&header, sizeof(header), 1, file);
    fwrite(model->conv_w, sizeof(f32), CNN_FILTERS * CNN_KERNEL * CNN_KERNEL, file);
    fwrite(model->conv_b, sizeof(f32), CNN_FILTERS, file);
    fwrite(model->fc1_w, sizeof(f32), CNN_HIDDEN_UNITS * CNN_FEATURES, file);
    fwrite(model->fc1_b, sizeof(f32), CNN_HIDDEN_UNITS, file);
    fwrite(model->fc2_w, sizeof(f32), MNIST_OUTPUT_SIZE * CNN_HIDDEN_UNITS, file);
    fwrite(model->fc2_b, sizeof(f32), MNIST_OUTPUT_SIZE, file);

    fclose(file);
    return true;
}

static b32 load_weights(cnn_model* model, const char* path) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        log_error("Failed to open %s\n", path);
        return false;
    }

    weights_header header = { 0 };
    if (fread(&header, sizeof(header), 1, file) != 1) {
        fclose(file);
        log_error("Failed to read weights header from %s\n", path);
        return false;
    }

    if (
        memcmp(header.magic, "DRCNN01", 7) != 0 ||
        header.version != 1 ||
        header.filters != CNN_FILTERS ||
        header.kernel != CNN_KERNEL ||
        header.hidden_units != CNN_HIDDEN_UNITS
    ) {
        fclose(file);
        log_error("Weights file %s does not match this CNN architecture.\n", path);
        return false;
    }

    if (
        fread(model->conv_w, sizeof(f32), CNN_FILTERS * CNN_KERNEL * CNN_KERNEL, file) != CNN_FILTERS * CNN_KERNEL * CNN_KERNEL ||
        fread(model->conv_b, sizeof(f32), CNN_FILTERS, file) != CNN_FILTERS ||
        fread(model->fc1_w, sizeof(f32), CNN_HIDDEN_UNITS * CNN_FEATURES, file) != CNN_HIDDEN_UNITS * CNN_FEATURES ||
        fread(model->fc1_b, sizeof(f32), CNN_HIDDEN_UNITS, file) != CNN_HIDDEN_UNITS ||
        fread(model->fc2_w, sizeof(f32), MNIST_OUTPUT_SIZE * CNN_HIDDEN_UNITS, file) != MNIST_OUTPUT_SIZE * CNN_HIDDEN_UNITS ||
        fread(model->fc2_b, sizeof(f32), MNIST_OUTPUT_SIZE, file) != MNIST_OUTPUT_SIZE
    ) {
        fclose(file);
        log_error("Weights file %s is truncated or invalid.\n", path);
        return false;
    }

    fclose(file);
    return true;
}

static void print_prediction(const char* label, const cnn_cache* cache) {
    printf(
        "%s (predicted=%u, confidence=%.2f%%): ",
        label,
        argmax10(cache->probs),
        cache->probs[argmax10(cache->probs)] * 100.0f
    );

    for (u32 i = 0; i < MNIST_OUTPUT_SIZE; i++) {
        printf("%.4f ", cache->probs[i]);
    }
    printf("\n");
}

static i32 run_train(const app_config* config) {
    mnist_split train = { 0 };
    mnist_split test = { 0 };
    cnn_model model = { 0 };
    cnn_grads grads = { 0 };
    cnn_cache cache = { 0 };
    cnn_cache eval_cache = { 0 };
    u32* order = NULL;
    i32 exit_code = 1;

    if (
        !load_mnist_split(config->data_dir, "train_images.mat", "train_labels.mat", MNIST_TRAIN_IMAGES, &train) ||
        !load_mnist_split(config->data_dir, "test_images.mat", "test_labels.mat", MNIST_TEST_IMAGES, &test)
    ) {
        goto cleanup;
    }

    if (
        !alloc_model(&model) ||
        !alloc_grads(&grads) ||
        !alloc_cache(&cache) ||
        !alloc_cache(&eval_cache)
    ) {
        fprintf(stderr, "Failed to allocate CNN state.\n");
        goto cleanup;
    }

    init_model(&model);

    order = (u32*)malloc(sizeof(u32) * train.count);
    if (order == NULL) {
        fprintf(stderr, "Failed to allocate training order.\n");
        goto cleanup;
    }

    for (u32 i = 0; i < train.count; i++) {
        order[i] = i;
    }

    if (config->sample_index < test.count) {
        const f32* sample = test.images + (u64)config->sample_index * MNIST_INPUT_SIZE;
        printf("Previewing test sample %u\n\n", config->sample_index);
        draw_mnist_digit(sample);
        printf("label: %u\n\n", test.labels[config->sample_index]);
        cnn_forward(&model, sample, &cache);
        print_prediction("Pre-training output", &cache);
    }

    u32 num_batches = (train.count + config->batch_size - 1) / config->batch_size;

    for (u32 epoch = 0; epoch < config->epochs; epoch++) {
        shuffle_indices(order, train.count);

        for (u32 batch = 0; batch < num_batches; batch++) {
            u32 batch_start = batch * config->batch_size;
            u32 batch_end = MIN(batch_start + config->batch_size, train.count);
            u32 batch_count = batch_end - batch_start;
            f32 batch_loss = 0.0f;

            zero_grads(&grads);

            for (u32 i = batch_start; i < batch_end; i++) {
                u32 index = order[i];
                const f32* image = train.images + (u64)index * MNIST_INPUT_SIZE;
                u8 label = train.labels[index];

                cnn_forward(&model, image, &cache);
                batch_loss += cnn_backward(&model, &grads, image, label, &cache);
            }

            apply_grads(&model, &grads, config->learning_rate / batch_count);

            printf(
                "Epoch %2u / %2u, Batch %4u / %4u, Average Loss: %.4f\r",
                epoch + 1, config->epochs,
                batch + 1, num_batches,
                batch_loss / batch_count
            );
            fflush(stdout);
        }

        printf("\n");

        u32 correct = 0;
        f32 avg_loss = 0.0f;
        evaluate_model(&model, &test, &eval_cache, &correct, &avg_loss);
        printf(
            "Test Completed. Accuracy: %5u / %5u (%.2f%%), Average Loss: %.4f\n",
            correct,
            test.count,
            (f32)correct / test.count * 100.0f,
            avg_loss
        );
    }

    if (config->sample_index < test.count) {
        const f32* sample = test.images + (u64)config->sample_index * MNIST_INPUT_SIZE;
        cnn_forward(&model, sample, &cache);
        print_prediction("Post-training output", &cache);
    }

    if (!save_weights(&model, config->weights_path)) {
        goto cleanup;
    }

    printf("Saved CNN weights to %s\n", config->weights_path);
    exit_code = 0;

cleanup:
    free(order);
    free_cache(&cache);
    free_cache(&eval_cache);
    free_model(&model);
    free_model((cnn_model*)&grads);
    free_mnist_split(&train);
    free_mnist_split(&test);
    return exit_code;
}

static i32 run_predict(const app_config* config) {
    mnist_split test = { 0 };
    cnn_model model = { 0 };
    cnn_cache cache = { 0 };
    i32 exit_code = 1;

    if (!load_mnist_split(config->data_dir, "test_images.mat", "test_labels.mat", MNIST_TEST_IMAGES, &test)) {
        goto cleanup;
    }

    if (config->sample_index >= test.count) {
        fprintf(stderr, "Sample index %u is out of range.\n", config->sample_index);
        goto cleanup;
    }

    if (!alloc_model(&model) || !alloc_cache(&cache)) {
        fprintf(stderr, "Failed to allocate prediction state.\n");
        goto cleanup;
    }

    if (!load_weights(&model, config->weights_path)) {
        goto cleanup;
    }

    const f32* sample = test.images + (u64)config->sample_index * MNIST_INPUT_SIZE;
    draw_mnist_digit(sample);
    printf("label: %u\n\n", test.labels[config->sample_index]);
    cnn_forward(&model, sample, &cache);
    print_prediction("CNN output", &cache);
    exit_code = 0;

cleanup:
    free_cache(&cache);
    free_model(&model);
    free_mnist_split(&test);
    return exit_code;
}

static char* load_text_file(const char* path, u64* out_size) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }

    long size = ftell(file);
    if (size < 0) {
        fclose(file);
        return NULL;
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }

    char* buffer = (char*)malloc((size_t)size + 1);
    if (buffer == NULL) {
        fclose(file);
        return NULL;
    }

    size_t read_size = fread(buffer, 1, (size_t)size, file);
    fclose(file);

    if (read_size != (size_t)size) {
        free(buffer);
        return NULL;
    }

    buffer[size] = '\0';
    if (out_size != NULL) {
        *out_size = (u64)size;
    }
    return buffer;
}

static void send_response(
    i32 client,
    const char* status,
    const char* content_type,
    const void* body,
    u64 body_size
) {
    char header[256] = { 0 };
    i32 header_size = snprintf(
        header,
        sizeof(header),
        "HTTP/1.1 %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %llu\r\n"
        "Connection: close\r\n\r\n",
        status,
        content_type,
        (unsigned long long)body_size
    );

    send(client, header, (size_t)header_size, 0);
    if (body != NULL && body_size > 0) {
        send(client, body, (size_t)body_size, 0);
    }
}

static void send_text(i32 client, const char* status, const char* text) {
    send_response(client, status, "text/plain; charset=utf-8", text, strlen(text));
}

static void send_json(i32 client, const char* json) {
    send_response(client, "200 OK", "application/json; charset=utf-8", json, strlen(json));
}

static void serve_asset(i32 client, const char* web_dir, const char* asset_name, const char* content_type) {
    char path[256] = { 0 };
    snprintf(path, sizeof(path), "%s/%s", web_dir, asset_name);

    u64 body_size = 0;
    char* body = load_text_file(path, &body_size);
    if (body == NULL) {
        send_text(client, "404 Not Found", "Asset not found.\n");
        return;
    }

    send_response(client, "200 OK", content_type, body, body_size);
    free(body);
}

static b32 parse_prediction_body(const char* body, f32* out_pixels) {
    u32 count = 0;
    const char* cursor = body;

    while (*cursor != '\0' && count < MNIST_INPUT_SIZE) {
        while (*cursor != '\0' && !(isdigit((unsigned char)*cursor) || *cursor == '-' || *cursor == '.')) {
            cursor++;
        }

        if (*cursor == '\0') {
            break;
        }

        char* end = NULL;
        f32 value = strtof(cursor, &end);
        if (end == cursor) {
            break;
        }

        if (value < 0.0f) value = 0.0f;
        if (value > 1.0f) value = 1.0f;
        out_pixels[count++] = value;
        cursor = end;
    }

    return count == MNIST_INPUT_SIZE;
}

static void handle_predict_request(i32 client, const cnn_model* model, const char* body) {
    f32 pixels[MNIST_INPUT_SIZE] = { 0 };
    cnn_cache cache = { 0 };
    char json[2048] = { 0 };
    i32 offset = 0;

    if (!parse_prediction_body(body, pixels)) {
        send_text(client, "400 Bad Request", "Expected 784 normalized pixel values.\n");
        return;
    }

    if (!alloc_cache(&cache)) {
        send_text(client, "500 Internal Server Error", "Failed to allocate inference cache.\n");
        return;
    }

    cnn_forward(model, pixels, &cache);

    offset += snprintf(
        json + offset,
        sizeof(json) - (size_t)offset,
        "{\"predicted\":%u,\"confidence\":%.6f,\"probabilities\":[",
        argmax10(cache.probs),
        cache.probs[argmax10(cache.probs)]
    );

    for (u32 i = 0; i < MNIST_OUTPUT_SIZE; i++) {
        offset += snprintf(
            json + offset,
            sizeof(json) - (size_t)offset,
            "%s%.6f",
            i == 0 ? "" : ",",
            cache.probs[i]
        );
    }

    snprintf(json + offset, sizeof(json) - (size_t)offset, "]}");
    send_json(client, json);
    free_cache(&cache);
}

static void handle_client(i32 client, const app_config* config, const cnn_model* model) {
    char* request = (char*)calloc(HTTP_BUFFER_SIZE, 1);
    if (request == NULL) {
        send_text(client, "500 Internal Server Error", "Out of memory.\n");
        return;
    }

    ssize_t received = recv(client, request, HTTP_BUFFER_SIZE - 1, 0);
    if (received <= 0) {
        free(request);
        return;
    }

    request[received] = '\0';

    char method[8] = { 0 };
    char path[128] = { 0 };
    sscanf(request, "%7s %127s", method, path);

    if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
        send_text(client, "200 OK", "ok\n");
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
        serve_asset(client, config->web_dir, "index.html", "text/html; charset=utf-8");
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/app.js") == 0) {
        serve_asset(client, config->web_dir, "app.js", "application/javascript; charset=utf-8");
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/styles.css") == 0) {
        serve_asset(client, config->web_dir, "styles.css", "text/css; charset=utf-8");
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/api/predict") == 0) {
        char* body = strstr(request, "\r\n\r\n");
        if (body == NULL) {
            send_text(client, "400 Bad Request", "Malformed HTTP request.\n");
        } else {
            body += 4;
            handle_predict_request(client, model, body);
        }
    } else {
        send_text(client, "404 Not Found", "Not found.\n");
    }

    free(request);
}

static i32 run_server(const app_config* config) {
    cnn_model model = { 0 };
    i32 server_fd = -1;
    i32 exit_code = 1;

    if (!alloc_model(&model)) {
        fprintf(stderr, "Failed to allocate CNN weights.\n");
        return 1;
    }

    if (!load_weights(&model, config->weights_path)) {
        fprintf(
            stderr,
            "A trained weights file is required before serving.\n"
            "Run `./digitrecg train --data-dir %s --weights %s` first.\n",
            config->data_dir,
            config->weights_path
        );
        goto cleanup;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        goto cleanup;
    }

    i32 reuse = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(config->port);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        perror("bind");
        goto cleanup;
    }

    if (listen(server_fd, 16) != 0) {
        perror("listen");
        goto cleanup;
    }

    printf("CNN web server listening on http://127.0.0.1:%u\n", config->port);
    printf("Press Ctrl+C to stop.\n");

    while (g_keep_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        i32 client = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client < 0) {
            if (errno == EINTR && !g_keep_running) {
                break;
            }
            perror("accept");
            continue;
        }

        handle_client(client, config, &model);
        close(client);
    }

    exit_code = 0;

cleanup:
    if (server_fd >= 0) {
        close(server_fd);
    }
    free_model(&model);
    return exit_code;
}

int main(i32 argc, char** argv) {
    app_config config = {
        .mode = APP_MODE_TRAIN,
        .data_dir = "data",
        .weights_path = "model_weights.bin",
        .web_dir = "web",
        .epochs = 2,
        .batch_size = 64,
        .learning_rate = 0.01f,
        .sample_index = 7,
        .port = 8080,
    };

    parse_args_result parse_result = parse_args(argc, argv, &config);
    if (parse_result == PARSE_ARGS_HELP) {
        return 0;
    }
    if (parse_result == PARSE_ARGS_ERROR) {
        return 1;
    }

    prng_seed(0xC0FFEEULL, 0xBADC0DEULL);

    switch (config.mode) {
        case APP_MODE_TRAIN: return run_train(&config);
        case APP_MODE_PREDICT: return run_predict(&config);
        case APP_MODE_SERVE: return run_server(&config);
    }

    return 1;
}
