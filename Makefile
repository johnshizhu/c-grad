CC     = gcc
CFLAGS = -Wall -Wextra -g
INC    = -I include/
SRC    = $(wildcard src/*.c)
TESTS  = $(patsubst tests/%.c, build/%, $(wildcard tests/*.c))

ifeq ($(OS), Windows_NT)
    MKDIR = mkdir
    RMDIR = rmdir /S /Q
else
    MKDIR = mkdir -p
    RMDIR = rm -rf
endif

build/%: tests/%.c $(SRC) | build
	$(CC) $(CFLAGS) $(INC) -o $@ $^

build:
	$(MKDIR) build

clean:
	$(RMDIR) build