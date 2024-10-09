CC = gcc

SRC_DIR = src
INCLUDE_DIR = include
# LIB_DIR = lib

# LIBS = -L$(LIB_DIR)
CFLAGS = -I$(INCLUDE_DIR) -Wall -g

SRC_FILES = $(SRC_DIR)/main.c \
			$(SRC_DIR)/image_IO.c \
			$(SRC_DIR)/image_modification.c \
			$(SRC_DIR)/image_recognition.c

OUTPUT = main

all: $(OUTPUT)

$(OUTPUT): $(SRC_FILES)
	$(CC) $(CFLAGS) $(SRC_FILES) -o $(OUTPUT)

clean:
	rm -f $(OUTPUT)