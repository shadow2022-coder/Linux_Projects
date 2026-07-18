CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -pedantic
LDFLAGS ?=
LDLIBS ?= -lm

TARGET := digitrecg
SRC := main.c arena.c prng.c
OBJ := $(SRC:.c=.o)
DATA_DIR ?= data
PACKAGE_DIR := packaged/digit-recognizer-c

.PHONY: all clean run prepare-data package

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) $(LDLIBS) -o $@

%.o: %.c base.h arena.h prng.h
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) --data-dir $(DATA_DIR)

prepare-data:
	python3 mnist.py --output-dir $(DATA_DIR)

package:
	mkdir -p $(PACKAGE_DIR)
	cp README.md Makefile .gitignore base.h arena.h arena.c prng.h prng.c main.c mnist.py $(PACKAGE_DIR)/

clean:
	rm -f $(TARGET) $(OBJ)
