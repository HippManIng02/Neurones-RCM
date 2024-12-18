CC = g++
CFLAGS = -Wall -g

all: main

main: main.o Mlp.o mnist_reader.o
	$(CC) $(CFLAGS) -o main main.o Mlp.o mnist_reader.o -lm

main.o: main.cpp mnist_reader.h Mlp.h
	$(CC) $(CFLAGS) -c main.cpp

Mlp.o: Mlp.cpp Mlp.h
	$(CC) $(CFLAGS) -c Mlp.cpp

mnist_reader.o: mnist_reader.c mnist_reader.h
	$(CC) $(CFLAGS) -c mnist_reader.c

clean:
	rm -f *.o main
