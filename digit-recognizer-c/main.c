
#define _CRT_SECURE_NO_WARNINGS

#include "base.h"
#include "arena.h"
#include "prng.h"

#include <errno.h>
#include <stdlib.h>

typedef struct {
    u32 rows, cols;
    // Row-major
    f32* data;
} matrix;

enum {
    MNIST_IMAGE_SIDE = 28,
    MNIST_INPUT_SIZE = 28 * 28,
    MNIST_OUTPUT_SIZE = 10,
    MNIST_TRAIN_IMAGES = 60000,
    MNIST_TEST_IMAGES = 10000,
};

typedef struct {
    const char* data_dir;
    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
    u32 preview_index;
} app_config;

typedef enum {
    PARSE_ARGS_OK = 0,
    PARSE_ARGS_HELP,
    PARSE_ARGS_ERROR,
} parse_args_result;

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);
matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename);
b32 mat_copy(matrix* dst, matrix* src);
void mat_clear(matrix* mat);
void mat_fill(matrix* mat, f32 x);
void mat_fill_rand(matrix* mat, f32 lower, f32 upper);
void mat_scale(matrix* mat, f32 scale);
f32 mat_sum(matrix* mat);
u64 mat_argmax(matrix* mat);
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);
b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad);
b32 mat_softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
);
b32 mat_cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad
);

typedef enum {
    MV_FLAG_NONE = 0,

    MV_FLAG_REQUIRES_GRAD  = (1 << 0),
    MV_FLAG_PARAMETER      = (1 << 1),
    MV_FLAG_INPUT          = (1 << 2),
    MV_FLAG_OUTPUT         = (1 << 3),
    MV_FLAG_DESIRED_OUTPUT = (1 << 4),
    MV_FLAG_COST           = (1 << 5),
} model_var_flags;

typedef enum {
    MV_OP_NULL = 0,
    MV_OP_CREATE,

    _MV_OP_UNARY_START,

    MV_OP_RELU,
    MV_OP_SOFTMAX,

    _MV_OP_BINARY_START,

    MV_OP_ADD,
    MV_OP_SUB,
    MV_OP_MATMUL,
    MV_OP_CROSS_ENTROPY,
} model_var_op;

#define MODEL_VAR_MAX_INPUTS 2
#define MV_NUM_INPUTS(op) ((op) < _MV_OP_UNARY_START ? 0 : ((op) < _MV_OP_BINARY_START ? 1 : 2))

typedef struct model_var {
    u32 index;
    u32 flags;

    matrix* val;
    matrix* grad;

    model_var_op op;
    struct model_var* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

typedef struct {
    model_var** vars;
    u32 size;
} model_program;

typedef struct {
    u32 num_vars;

    model_var* input;
    model_var* output;
    model_var* desired_output;
    model_var* cost;

    model_program forward_prog;
    model_program cost_prog;
} model_context;

typedef struct {
    matrix* train_images;
    matrix* train_labels;
    matrix* test_images;
    matrix* test_labels;

    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
} model_training_desc;

model_var* mv_create(
    mem_arena* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
);

model_var* mv_relu(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
);
model_var* mv_softmax(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
);

model_var* mv_add(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);
model_var* mv_sub(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);
model_var* mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);
model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* p, model_var* q, u32 flags
);

model_program model_prog_create(
    mem_arena* arena, model_context* model, model_var* out_var
);
void model_prog_compute(model_program* prog);
void model_prog_compute_grads(model_program* prog);

model_context* model_create(mem_arena* arena);
void model_compile(mem_arena* arena, model_context* model);
void model_feedforward(model_context* model);
void model_train(
    model_context* model,
    const model_training_desc* training_desc
);

void draw_mnist_digit(const f32* data);
void create_mnist_model(mem_arena* arena, model_context* model);
void print_usage(const char* argv0);
void print_prediction(const char* label, const matrix* output);
b32 parse_u32_arg(const char* text, u32* out_value);
b32 parse_f32_arg(const char* text, f32* out_value);
parse_args_result parse_args(i32 argc, char** argv, app_config* config);
b32 build_data_path(char* out, u64 out_size, const char* data_dir, const char* filename);
matrix* mat_load_one_hot_labels(
    mem_arena* arena, u32 rows, const char* filename
);

int main(i32 argc, char** argv) {
    app_config config = {
        .data_dir = "data",
        .epochs = 10,
        .batch_size = 50,
        .learning_rate = 0.01f,
        .preview_index = 0,
    };

    parse_args_result parse_result = parse_args(argc, argv, &config);
    if (parse_result == PARSE_ARGS_HELP) {
        return 0;
    }
    if (parse_result == PARSE_ARGS_ERROR) {
        return 1;
    }

    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));
    if (perm_arena == NULL) {
        fprintf(stderr, "Failed to allocate the main memory arena.\n");
        return 1;
    }

    char train_images_path[256] = { 0 };
    char train_labels_path[256] = { 0 };
    char test_images_path[256] = { 0 };
    char test_labels_path[256] = { 0 };

    if (
        !build_data_path(train_images_path, sizeof(train_images_path), config.data_dir, "train_images.mat") ||
        !build_data_path(train_labels_path, sizeof(train_labels_path), config.data_dir, "train_labels.mat") ||
        !build_data_path(test_images_path, sizeof(test_images_path), config.data_dir, "test_images.mat") ||
        !build_data_path(test_labels_path, sizeof(test_labels_path), config.data_dir, "test_labels.mat")
    ) {
        fprintf(stderr, "The dataset path is too long.\n");
        arena_destroy(perm_arena);
        return 1;
    }

    matrix* train_images = mat_load(
        perm_arena, MNIST_TRAIN_IMAGES, MNIST_INPUT_SIZE, train_images_path
    );
    matrix* test_images = mat_load(
        perm_arena, MNIST_TEST_IMAGES, MNIST_INPUT_SIZE, test_images_path
    );
    matrix* train_labels = mat_load_one_hot_labels(
        perm_arena, MNIST_TRAIN_IMAGES, train_labels_path
    );
    matrix* test_labels = mat_load_one_hot_labels(
        perm_arena, MNIST_TEST_IMAGES, test_labels_path
    );

    if (
        train_images == NULL || test_images == NULL ||
        train_labels == NULL || test_labels == NULL
    ) {
        fprintf(
            stderr,
            "MNIST data files are missing or invalid.\n"
            "Run `python3 mnist.py --output-dir %s` first.\n",
            config.data_dir
        );
        arena_destroy(perm_arena);
        return 1;
    }

    if (config.preview_index >= test_images->rows) {
        fprintf(
            stderr,
            "Preview index %u is out of range for %u test images.\n",
            config.preview_index,
            test_images->rows
        );
        arena_destroy(perm_arena);
        return 1;
    }

    u64 preview_offset = (u64)config.preview_index * MNIST_INPUT_SIZE;
    u64 preview_label_offset = (u64)config.preview_index * MNIST_OUTPUT_SIZE;

    printf("Previewing test sample %u from %s\n\n", config.preview_index, config.data_dir);
    draw_mnist_digit(test_images->data + preview_offset);
    printf("label: ");
    for (u32 i = 0; i < MNIST_OUTPUT_SIZE; i++) {
        printf("%.0f ", test_labels->data[preview_label_offset + i]);
    }
    printf("\n\n");

    model_context* model = model_create(perm_arena);
    create_mnist_model(perm_arena, model);
    model_compile(perm_arena, model);

    memcpy(
        model->input->val->data,
        test_images->data + preview_offset,
        sizeof(f32) * MNIST_INPUT_SIZE
    );
    model_feedforward(model);

    print_prediction("Pre-training output", model->output->val);

    model_training_desc training_desc = {
        .train_images = train_images,
        .train_labels = train_labels,
        .test_images = test_images,
        .test_labels = test_labels,

        .epochs = config.epochs,
        .batch_size = config.batch_size,
        .learning_rate = config.learning_rate
    };
    model_train(model, &training_desc);
    
    memcpy(
        model->input->val->data,
        test_images->data + preview_offset,
        sizeof(f32) * MNIST_INPUT_SIZE
    );
    model_feedforward(model);
    print_prediction("Post-training output", model->output->val);

    arena_destroy(perm_arena);

    return 0;
}

void draw_mnist_digit(const f32* data) {
    for (u32 y = 0; y < MNIST_IMAGE_SIDE; y++) {
        for (u32 x = 0; x < MNIST_IMAGE_SIDE; x++) {
            f32 num = data[x + y * MNIST_IMAGE_SIDE];
            u32 col = 232 + (u32)(num * 23);
            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\n");
    }
    printf("\x1b[0m");
}

void print_usage(const char* argv0) {
    printf(
        "Usage: %s [options]\n"
        "  --data-dir <path>        Directory containing MNIST .mat files (default: data)\n"
        "  --epochs <count>         Training epochs (default: 10)\n"
        "  --batch-size <count>     Mini-batch size (default: 50)\n"
        "  --learning-rate <value>  SGD learning rate (default: 0.01)\n"
        "  --sample-index <index>   Test image to preview before training (default: 0)\n"
        "  --help                   Show this help text\n",
        argv0
    );
}

void print_prediction(const char* label, const matrix* output) {
    printf(
        "%s (predicted=%llu): ",
        label,
        (unsigned long long)mat_argmax((matrix*)output)
    );
    for (u32 i = 0; i < output->rows * output->cols; i++) {
        printf("%.4f ", output->data[i]);
    }
    printf("\n");
}

b32 parse_u32_arg(const char* text, u32* out_value) {
    char* end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (text[0] == '\0' || end == text || *end != '\0' || value > UINT32_MAX) {
        return false;
    }

    *out_value = (u32)value;
    return true;
}

b32 parse_f32_arg(const char* text, f32* out_value) {
    char* end = NULL;
    errno = 0;
    float value = strtof(text, &end);
    if (text[0] == '\0' || end == text || *end != '\0' || errno != 0) {
        return false;
    }

    *out_value = value;
    return true;
}

parse_args_result parse_args(i32 argc, char** argv, app_config* config) {
    for (i32 i = 1; i < argc; i++) {
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
        } else if (strcmp(arg, "--epochs") == 0) {
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &config->epochs) || config->epochs == 0) {
                fprintf(stderr, "--epochs requires a positive integer.\n");
                return PARSE_ARGS_ERROR;
            }
        } else if (strcmp(arg, "--batch-size") == 0) {
            if (
                i + 1 >= argc ||
                !parse_u32_arg(argv[++i], &config->batch_size) ||
                config->batch_size == 0
            ) {
                fprintf(stderr, "--batch-size requires a positive integer.\n");
                return PARSE_ARGS_ERROR;
            }
        } else if (strcmp(arg, "--learning-rate") == 0) {
            if (
                i + 1 >= argc ||
                !parse_f32_arg(argv[++i], &config->learning_rate) ||
                config->learning_rate <= 0.0f
            ) {
                fprintf(stderr, "--learning-rate requires a positive number.\n");
                return PARSE_ARGS_ERROR;
            }
        } else if (strcmp(arg, "--sample-index") == 0) {
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &config->preview_index)) {
                fprintf(stderr, "--sample-index requires a non-negative integer.\n");
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

b32 build_data_path(char* out, u64 out_size, const char* data_dir, const char* filename) {
    i32 written = snprintf(out, (size_t)out_size, "%s/%s", data_dir, filename);
    return written > 0 && (u64)written < out_size;
}

matrix* mat_load_one_hot_labels(
    mem_arena* arena, u32 rows, const char* filename
) {
    matrix* labels_raw = mat_load(arena, rows, 1, filename);
    if (labels_raw == NULL) {
        return NULL;
    }

    matrix* labels = mat_create(arena, rows, MNIST_OUTPUT_SIZE);
    if (labels == NULL) {
        return NULL;
    }

    for (u32 i = 0; i < rows; i++) {
        u32 label = (u32)labels_raw->data[i];
        if (label >= MNIST_OUTPUT_SIZE) {
            fprintf(stderr, "Invalid label %u in %s at row %u.\n", label, filename, i);
            return NULL;
        }
        labels->data[i * MNIST_OUTPUT_SIZE + label] = 1.0f;
    }

    return labels;
}

void create_mnist_model(mem_arena* arena, model_context* model) {
    model_var* input = mv_create(arena, model, MNIST_INPUT_SIZE, 1, MV_FLAG_INPUT);

    model_var* W0 = mv_create(arena, model, 16, MNIST_INPUT_SIZE, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* W1 = mv_create(arena, model, 16, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* W2 = mv_create(arena, model, MNIST_OUTPUT_SIZE, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    f32 bound0 = sqrtf(6.0f / (MNIST_INPUT_SIZE + 16));
    f32 bound1 = sqrtf(6.0f / (16 + 16));
    f32 bound2 = sqrtf(6.0f / (16 + MNIST_OUTPUT_SIZE));
    mat_fill_rand(W0->val, -bound0, bound0);
    mat_fill_rand(W1->val, -bound1, bound1);
    mat_fill_rand(W2->val, -bound2, bound2);

    model_var* b0 = mv_create(arena, model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* b1 = mv_create(arena, model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* b2 = mv_create(arena, model, MNIST_OUTPUT_SIZE, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    model_var* z0_a = mv_matmul(arena, model, W0, input, 0);
    model_var* z0_b = mv_add(arena, model, z0_a, b0, 0);
    model_var* a0 = mv_relu(arena, model, z0_b, 0);

    model_var* z1_a = mv_matmul(arena, model, W1, a0, 0);
    model_var* z1_b = mv_add(arena, model, z1_a, b1, 0);
    model_var* z1_c = mv_relu(arena, model, z1_b, 0);
    model_var* a1 = mv_add(arena, model, a0, z1_c, 0);

    model_var* z2_a = mv_matmul(arena, model, W2, a1, 0);
    model_var* z2_b = mv_add(arena, model, z2_a, b2, 0);
    model_var* output = mv_softmax(arena, model, z2_b, MV_FLAG_OUTPUT);

    model_var* y = mv_create(arena, model, MNIST_OUTPUT_SIZE, 1, MV_FLAG_DESIRED_OUTPUT);

    (void)mv_cross_entropy(arena, model, y, output, MV_FLAG_COST);
}

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols) {
    matrix* mat = PUSH_STRUCT(arena, matrix);

    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);

    return mat;
}

matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename) {
    if (arena == NULL || filename == NULL) {
        return NULL;
    }

    matrix* mat = mat_create(arena, rows, cols);
    if (mat == NULL) {
        return NULL;
    }

    FILE* f = fopen(filename, "rb");
    if (f == NULL) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        fprintf(stderr, "Failed to seek %s\n", filename);
        return NULL;
    }

    long file_size = ftell(f);
    if (file_size < 0) {
        fclose(f);
        fprintf(stderr, "Failed to determine the size of %s\n", filename);
        return NULL;
    }

    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        fprintf(stderr, "Failed to rewind %s\n", filename);
        return NULL;
    }

    u64 expected_size = sizeof(f32) * (u64)rows * cols;
    if ((u64)file_size < expected_size) {
        fclose(f);
        fprintf(
            stderr,
            "%s is too small. Expected at least %llu bytes, got %ld.\n",
            filename,
            (unsigned long long)expected_size,
            file_size
        );
        return NULL;
    }

    size_t read_size = fread(mat->data, 1, (size_t)expected_size, f);
    if (read_size != (size_t)expected_size) {
        fclose(f);
        fprintf(stderr, "Failed to read %s completely.\n", filename);
        return NULL;
    }

    fclose(f);

    return mat;
}

b32 mat_copy(matrix* dst, matrix* src) {
    if (dst->rows != src->rows || dst->cols != src->cols) {
        return false;
    }

    memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);

    return true;
}

void mat_clear(matrix* mat) {
    memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}

void mat_fill(matrix* mat, f32 x) {
    u64 size = (u64)mat->rows * mat->cols;

    for (u64 i = 0; i < size; i++) {
        mat->data[i] = x;
    }
}

void mat_fill_rand(matrix* mat, f32 lower, f32 upper) {
    u64 size = (u64)mat->rows * mat->cols;

    for (u64 i = 0; i < size; i++) {
        mat->data[i] = prng_randf() * (upper - lower) + lower;
    }

}

void mat_scale(matrix* mat, f32 scale) {
    u64 size = (u64)mat->rows * mat->cols;

    for (u64 i = 0; i < size; i++) {
        mat->data[i] *= scale;
    }
}

f32 mat_sum(matrix* mat) {
    u64 size = (u64)mat->rows * mat->cols;

    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        sum += mat->data[i];
    }

    return sum;
}

u64 mat_argmax(matrix* mat) {
    u64 size = (u64)mat->rows * mat->cols;

    u64 max_i = 0;
    for (u64 i = 0; i < size; i++) {
        if (mat->data[i] > mat->data[max_i]) {
            max_i = i;
        }
    }

    return max_i;
}

b32 mat_add(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    if (out->rows != a->rows || out->cols != a->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

b32 mat_sub(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    if (out->rows != a->rows || out->cols != a->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

void _mat_mul_nn(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 k = 0; k < a->cols; k++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] += 
                    a->data[k + i * a->cols] * 
                    b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_nt(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->cols; k++) {
                out->data[j + i * out->cols] += 
                    a->data[k + i * a->cols] * 
                    b->data[k + j * b->cols];
            }
        }
    }
}

void _mat_mul_tn(matrix* out, const matrix* a, const matrix* b) {
    for (u64 k = 0; k < a->rows; k++) {
        for (u64 i = 0; i < out->rows; i++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] += 
                    a->data[i + k * a->cols] * 
                    b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_tt(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->rows; k++) {
                out->data[j + i * out->cols] += 
                    a->data[i + k * a->cols] * 
                    b->data[k + j * b->cols];
            }
        }
    }
}

b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
) {
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows) { return false; }
    if (out->rows != a_rows || out->cols != b_cols) { return false; }

    if (zero_out) {
        mat_clear(out);
    }

    u32 transpose = (transpose_a << 1) | transpose_b;
    switch (transpose) {
        case 0: { _mat_mul_nn(out, a, b); } break;
        case 1: { _mat_mul_nt(out, a, b); } break;
        case 2: { _mat_mul_tn(out, a, b); } break;
        case 3: { _mat_mul_tt(out, a, b); } break;
    }

    return true;
}

b32 mat_relu(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = MAX(0, in->data[i]);
    }

    return true;
}

b32 mat_softmax(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;

    f32 max_val = in->data[0];
    for (u64 i = 1; i < size; i++) {
        if (in->data[i] > max_val) {
            max_val = in->data[i];
        }
    }

    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i] - max_val);
        sum += out->data[i];
    }

    if (sum <= 0.0f) {
        return false;
    }

    mat_scale(out, 1.0f / sum);

    return true;
}

b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q) {
    if (p->rows != q->rows || p->cols != q->cols) { return false; }
    if (out->rows != p->rows || out->cols != p->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        f32 q_value = q->data[i] > 1e-7f ? q->data[i] : 1e-7f;
        out->data[i] = p->data[i] == 0.0f ?
            0.0f : p->data[i] * -logf(q_value);
    }

    return true;
}

b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }
    if (out->rows != grad->rows || out->cols != grad->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
    }

    return true;
}

b32 mat_softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
) {
    if (softmax_out->rows != 1 && softmax_out->cols != 1) {
        return false;
    }

    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    u32 size = MAX(softmax_out->rows, softmax_out->cols);
    matrix* jacobian = mat_create(scratch.arena, size, size);

    for (u32 i = 0; i < size; i++) {
        for (u32 j = 0; j < size; j++) {
            jacobian->data[j + i * size] =
                softmax_out->data[i] * ((i == j) - softmax_out->data[j]);
        }
    }

    mat_mul(out, jacobian, grad, 0, 0, 0);

    arena_scratch_release(scratch);

    return true;
}

b32 mat_cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad
) {
    if (p->rows != q->rows || p->cols != q->cols) { return false; }

    u64 size = (u64)p->rows * p->cols;

    if (p_grad != NULL) {
        if (p_grad->rows != p->rows || p_grad->cols != p->cols) {
            return false; 
        }

        for (u64 i = 0; i < size; i++) {
            f32 q_value = q->data[i] > 1e-7f ? q->data[i] : 1e-7f;
            p_grad->data[i] += -logf(q_value) * grad->data[i];
        }
    }

    if (q_grad != NULL) {
        if (q_grad->rows != q->rows || q_grad->cols != q->cols) {
            return false; 
        }

        for (u64 i = 0; i < size; i++) {
            f32 q_value = q->data[i] > 1e-7f ? q->data[i] : 1e-7f;
            q_grad->data[i] += -p->data[i] / q_value * grad->data[i];
        }
    }

    return true;
}

model_var* mv_create(
    mem_arena* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
) {
    model_var* out = PUSH_STRUCT(arena, model_var);

    out->index = model->num_vars++;
    out->flags = flags;
    out->op = MV_OP_CREATE;
    out->val = mat_create(arena, rows, cols);

    if (flags & MV_FLAG_REQUIRES_GRAD) {
        out->grad = mat_create(arena, rows, cols);
    }

    if (flags & MV_FLAG_INPUT) { model->input = out; }
    if (flags & MV_FLAG_OUTPUT) { model->output = out; }
    if (flags & MV_FLAG_DESIRED_OUTPUT) { model->desired_output = out; }
    if (flags & MV_FLAG_COST) { model->cost = out; }

    return out;
}

model_var* _mv_unary_impl(
    mem_arena* arena, model_context* model,
    model_var* input, u32 rows, u32 cols,
    u32 flags, model_var_op op
) {
    if (input->flags & MV_FLAG_REQUIRES_GRAD) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var* out = mv_create(arena, model, rows, cols, flags);

    out->op = op;
    out->inputs[0] = input;

    return out;
}

model_var* _mv_binary_impl(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b,
    u32 rows, u32 cols,
    u32 flags, model_var_op op
) {
    if (
        (a->flags & MV_FLAG_REQUIRES_GRAD) ||
        (b->flags & MV_FLAG_REQUIRES_GRAD)
    ) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var* out = mv_create(arena, model, rows, cols, flags);

    out->op = op;
    out->inputs[0] = a;
    out->inputs[1] = b;

    return out;
}

model_var* mv_relu(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
) {
    return _mv_unary_impl(
        arena, model, input,
        input->val->rows, input->val->cols,
        flags, MV_OP_RELU
    );
}

model_var* mv_softmax(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
) {
    return _mv_unary_impl(
        arena, model, input,
        input->val->rows, input->val->cols,
        flags, MV_OP_SOFTMAX
    );
}

model_var* mv_add(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }

    return _mv_binary_impl(
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_ADD
    );
}

model_var* mv_sub(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }

    return _mv_binary_impl(
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_SUB
    );
}

model_var* mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    if (a->val->cols != b->val->rows) {
        return NULL;
    }

    return _mv_binary_impl(
        arena, model, a, b,
        a->val->rows, b->val->cols,
        flags, MV_OP_MATMUL
    );
}

model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* p, model_var* q, u32 flags
) {
    if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
        return NULL;
    }

    return _mv_binary_impl(
        arena, model, p, q,
        p->val->rows, p->val->cols,
        flags, MV_OP_CROSS_ENTROPY
    );
}

model_program model_prog_create(
    mem_arena* arena, model_context* model, model_var* out_var
) {
    mem_arena_temp scratch = arena_scratch_get(&arena, 1);

    b8* visited = PUSH_ARRAY(scratch.arena, b8, model->num_vars);

    u32 stack_size = 0;
    u32 out_size = 0;
    model_var** stack = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);
    model_var** out = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);

    stack[stack_size++] = out_var;

    while (stack_size > 0) {
        model_var* cur = stack[--stack_size];

        if (cur->index >= model->num_vars) { continue; }

        if (visited[cur->index]) {
            if (out_size < model->num_vars) {
                out[out_size++] = cur;
            }
            continue;
        }

        visited[cur->index] = true;

        if (stack_size < model->num_vars) {
            stack[stack_size++] = cur;
        }

        u32 num_inputs = MV_NUM_INPUTS(cur->op);
        for (u32 i = 0; i < num_inputs; i++) {
            model_var* input = cur->inputs[i];

            if (input->index >= model->num_vars || visited[input->index]) {
                continue;
            }

            for (u32 j = 0; j < stack_size; j++) {
                if (stack[j] == input) {
                    for (u32 k = j; k < stack_size-1; k++) {
                        stack[k] = stack[k+1];
                    }
                    stack_size--;
                }
            }

            if (stack_size < model->num_vars) {
                stack[stack_size++] = input;
            }
        }
    }

    model_program prog = {
        .size = out_size,
        .vars = PUSH_ARRAY_NZ(arena, model_var*, out_size)
    };

    memcpy(prog.vars, out, sizeof(model_var*) * out_size);

    arena_scratch_release(scratch);

    return prog;
}

void model_prog_compute(model_program* prog) {
    for (u32 i = 0; i < prog->size; i++) {
        model_var* cur = prog->vars[i];

        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];

        switch (cur->op) {
            case MV_OP_NULL:
            case MV_OP_CREATE: break;

            case _MV_OP_UNARY_START: break;

            case MV_OP_RELU: { mat_relu(cur->val, a->val); } break;
            case MV_OP_SOFTMAX: { mat_softmax(cur->val, a->val); } break;

            case _MV_OP_BINARY_START: break;

            case MV_OP_ADD: { mat_add(cur->val, a->val, b->val); } break;
            case MV_OP_SUB: { mat_sub(cur->val, a->val, b->val); } break;
            case MV_OP_MATMUL: {
                mat_mul(cur->val, a->val, b->val, 1, 0, 0); 
            } break;
            case MV_OP_CROSS_ENTROPY: {
                mat_cross_entropy(cur->val, a->val, b->val);
            } break;
        }
    }
}

void model_prog_compute_grads(model_program* prog) {
    for (u32 i = 0; i < prog->size; i++) {
        model_var* cur = prog->vars[i];

        if ((cur->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) {
            continue;
        }

        if (cur->flags & MV_FLAG_PARAMETER) {
            continue;
        }

        mat_clear(cur->grad);
    }

    mat_fill(prog->vars[prog->size-1]->grad, 1.0f);

    for (i64 i = (i64)prog->size - 1; i >= 0; i--) {
        model_var* cur = prog->vars[i];

        if ((cur->flags & MV_FLAG_REQUIRES_GRAD) == 0) {
            continue;
        }

        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];

        u32 num_inputs = MV_NUM_INPUTS(cur->op);

        if (
            num_inputs == 1 &&
            (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD
        ) {
            continue;
        }

        if (
            num_inputs == 2 &&
            (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD && 
            (b->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD
        ) {
            continue;
        }

        switch (cur->op) {
            case MV_OP_NULL:
            case MV_OP_CREATE: break;

            case _MV_OP_UNARY_START: break;

            case MV_OP_RELU: {
                mat_relu_add_grad(a->grad, a->val, cur->grad);
            } break;
            case MV_OP_SOFTMAX: {
                mat_softmax_add_grad(a->grad, cur->val, cur->grad);
            } break;

            case _MV_OP_BINARY_START: break;

            case MV_OP_ADD: {
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(a->grad, a->grad, cur->grad);
                }

                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(b->grad, b->grad, cur->grad);
                }
            } break;

            case MV_OP_SUB: {
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(a->grad, a->grad, cur->grad);
                }

                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_sub(b->grad, b->grad, cur->grad);
                }
            } break;

            case MV_OP_MATMUL: {
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
                }

                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
                }
            } break;

            case MV_OP_CROSS_ENTROPY: {
                model_var* p = a;
                model_var* q = b;

                mat_cross_entropy_add_grad(
                    p->grad, q->grad, p->val, q->val, cur->grad
                );
            } break;
        }
    }
}

model_context* model_create(mem_arena* arena) {
    model_context* model = PUSH_STRUCT(arena, model_context);

    return model;
}

void model_compile(mem_arena* arena, model_context* model) {
    if (model->output != NULL) {
        model->forward_prog = model_prog_create(arena, model, model->output);
    }

    if (model->cost != NULL) {
        model->cost_prog = model_prog_create(arena, model, model->cost);
    }
}

void model_feedforward(model_context* model) {
    model_prog_compute(&model->forward_prog);
}

void model_train(
    model_context* model,
    const model_training_desc* training_desc
) {
    matrix* train_images = training_desc->train_images;
    matrix* train_labels = training_desc->train_labels;
    matrix* test_images = training_desc->test_images;
    matrix* test_labels = training_desc->test_labels;

    u32 num_examples = train_images->rows;
    u32 input_size = train_images->cols;
    u32 output_size = train_labels->cols;
    u32 num_tests = test_images->rows;

    u32 num_batches =
        (num_examples + training_desc->batch_size - 1) /
        training_desc->batch_size;

    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    u32* training_order = PUSH_ARRAY_NZ(scratch.arena, u32, num_examples);
    for (u32 i = 0; i < num_examples; i++) {
        training_order[i] = i;
    }

    for (u32 epoch = 0; epoch < training_desc->epochs; epoch++) {
        for (u32 i = 0; i < num_examples; i++) {
            u32 a = prng_rand() % num_examples;
            u32 b = prng_rand() % num_examples;

            u32 tmp = training_order[b];
            training_order[b] = training_order[a];
            training_order[a] = tmp;
        }

        for (u32 batch = 0; batch < num_batches; batch++) {
            u32 batch_start = batch * training_desc->batch_size;
            u32 batch_end = MIN(batch_start + training_desc->batch_size, num_examples);
            u32 batch_count = batch_end - batch_start;

            for (u32 i = 0; i < model->cost_prog.size; i++) {
                model_var* cur = model->cost_prog.vars[i];

                if (cur->flags & MV_FLAG_PARAMETER) {
                    mat_clear(cur->grad);
                }
            }

            f32 avg_cost = 0.0f;
            for (u32 i = 0; i < batch_count; i++) {
                u32 order_index = batch_start + i;
                u32 index = training_order[order_index];

                memcpy(
                    model->input->val->data,
                    train_images->data + index * input_size,
                    sizeof(f32) * input_size
                );

                memcpy(
                    model->desired_output->val->data,
                    train_labels->data + index * output_size,
                    sizeof(f32) * output_size
                );

                model_prog_compute(&model->cost_prog);
                model_prog_compute_grads(&model->cost_prog);

                avg_cost += mat_sum(model->cost->val);
            }
            avg_cost /= (f32)batch_count;

            for (u32 i = 0; i < model->cost_prog.size; i++) {
                model_var* cur = model->cost_prog.vars[i];

                if ((cur->flags & MV_FLAG_PARAMETER) != MV_FLAG_PARAMETER) {
                    continue;
                }

                mat_scale(
                    cur->grad,
                    training_desc->learning_rate /
                    batch_count
                );
                mat_sub(cur->val, cur->val, cur->grad);
            }

            printf(
                "Epoch %2d / %2d, Batch %4d / %4d, Average Cost: %.4f\r",
                epoch + 1, training_desc->epochs,
                batch + 1, num_batches, avg_cost
            );
            fflush(stdout);
        }
        printf("\n");

        u32 num_correct = 0;
        f32 avg_cost = 0;
        for (u32 i = 0; i < num_tests; i++) {
            memcpy(
                model->input->val->data,
                test_images->data + i * input_size,
                sizeof(f32) * input_size
            );

            memcpy(
                model->desired_output->val->data,
                test_labels->data + i * output_size,
                sizeof(f32) * output_size
            );

            model_prog_compute(&model->cost_prog);

            avg_cost += mat_sum(model->cost->val);
            num_correct +=
                mat_argmax(model->output->val) ==
                mat_argmax(model->desired_output->val);
        }

        avg_cost /= (f32)num_tests;
        printf(
            "Test Completed. Accuracy: %5d / %5d (%.1f%%), Average Cost: %.4f\n",
            num_correct, num_tests, (f32)num_correct / num_tests * 100.0f,
            avg_cost
        );
    }

    arena_scratch_release(scratch);
}
