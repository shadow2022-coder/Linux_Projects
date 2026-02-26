# Makefile for Thermal Monitor (raylib + raygui)
#
# Usage:
#   make            — build
#   make run        — build & run
#   make clean      — remove binary

CC      = gcc
TARGET  = thermal_monitor
SRC     = thermal_monitor.c
CFLAGS  = -O2 -Wall -Wextra
LIBS    = -lraylib -lm -lpthread -ldl -lrt -lX11

all: $(TARGET)

$(TARGET): $(SRC) raygui.h
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS)

run: all
	./$(TARGET)

# Download raygui header if missing
raygui.h:
	@echo "Downloading raygui.h ..."
	wget -q https://raw.githubusercontent.com/raysan5/raygui/master/src/raygui.h -O raygui.h

clean:
	rm -f $(TARGET)

.PHONY: all run clean