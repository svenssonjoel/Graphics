
TARGET = dungeon

C99 = gcc -std=c99 

LIBS = -lglut -lGLEW -lGLU -lGL -lm -l3ds -lpng
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all:  $(TARGET)

$(TARGET): $(TARGET).c 
	$(C99) $(CFLAGS) -o $(TARGET) $(TARGET).c $(LIBS) $(INC) $(GFX)


clean: 
	rm $(TARGET)