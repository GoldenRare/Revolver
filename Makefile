ifeq ($(OS), Windows_NT)
	EXECUTABLE = Revolver.exe
else
	EXECUTABLE = Revolver
endif

SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)

CC = gcc
CFLAGS = -std=c23 -pedantic -Wall -Wextra -Wshadow -Wcast-qual -static -O3 -march=native -flto
LDFLAGS = $(CFLAGS)

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<
	
clean:
	rm -f $(EXECUTABLE) $(OBJECTS) 