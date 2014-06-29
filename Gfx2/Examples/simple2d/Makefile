C99 = gcc -std=c99 

LIBS = -lglut -lGLEW -lGL -lGLU  -l3ds -lpng -lm
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all: simple

simple: simple.c 
	$(C99) $(CFLAGS) -o simple simple.c $(INC) $(GFX) $(LIBS)


clean: 
	rm simple

