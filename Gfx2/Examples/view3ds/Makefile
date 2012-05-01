C99 = gcc -std=c99 

LIBS = -lglut -lGLEW -lGLU -lm -l3ds -lpng
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all: view3ds

view3ds: view3ds.c 
	$(C99) $(CFLAGS) -o view3ds view3ds.c $(LIBS) $(INC) $(GFX)


clean: 
	rm view3ds	
