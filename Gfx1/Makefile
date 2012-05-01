

C99 = gcc -std=c99

LIBS = -lglut -lGLU -lGLEW

SW = -msse3

My3DLib = vector.o matrix.o shaders.o error.o


main: main.c $(My3DLib)
	$(C99) -o main main.c vector.o matrix.o $(LIBS)

gl32test: gl32test.c $(My3DLib) 
	$(C99) -o gl32test gl32test.c $(My3DLib) $(LIBS)

tridemo: tridemo.c $(My3DLib)
	$(C99) -o tridemo tridemo.c $(My3DLib) $(LIBS)


vector.o: vector.c 
	$(C99) -o vector.o -c vector.c


matrix.o: matrix.c
	$(C99) -o matrix.o -c matrix.c

shaders.o: shaders.c
	$(C99) -o shaders.o -c shaders.c

error.o: error.c
	$(C99) -o error.o -c error.c

all: main gl32test


clean: 
	rm main
	rm gl32test
	rm *.o