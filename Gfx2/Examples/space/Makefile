C99 = gcc -std=c99 

LIBS = -lglut -lGLEW -lGLU -lm -l3ds -lpng
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all: space

space: space.c 
	$(C99) $(CFLAGS) -o space space.c $(LIBS) $(INC) $(GFX)


clean: 
	rm space