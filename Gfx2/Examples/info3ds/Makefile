C99 = gcc -std=c99 

LIBS = -lGL -lGLU -lglut -lGLEW -lm -l3ds -lpng
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all: info3ds

info3ds: info3ds.c 
	$(C99) $(CFLAGS) -o info3ds info3ds.c $(GFX) $(LIBS) $(INC) 


clean: 
	rm info3ds
