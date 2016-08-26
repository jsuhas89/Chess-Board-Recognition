CPPFLAGS = $(shell  pkg-config --cflags opencv)
LDLIBS = $(shell pkg-config --libs opencv)

main1: main.cpp
	g++  -g $(CPPFLAGS) $(LDLIBS) -o main main.cpp
#pkg-config --cflags --libs opencv
