CC = mpicc
CFLAGS = -O2 -Wall
TARGET = ex3_1
SRC = ex3_1.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)