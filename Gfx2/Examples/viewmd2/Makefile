C99 = gcc -std=c99 

LIBS = -lglut -lGLEW -lGL -lGLU -lm -l3ds -lpng
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all:  viewmd2

viewmd2: viewmd2.c 
	$(C99) $(CFLAGS) -o viewmd2 viewmd2.c $(GFX) $(LIBS) $(INC) 


clean: 
	rm viewmd2
