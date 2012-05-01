C99 = gcc -std=c99 

LIBS = -lglut -lGLEW -lGLU -lm -l3ds -lpng
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all: simple

simple: simple.c 
	$(C99) $(CFLAGS) -o simple simple.c $(LIBS) $(INC) $(GFX)


clean: 
	rm simple

