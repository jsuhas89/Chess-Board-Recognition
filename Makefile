all: CImg.h final.cpp
	g++ final.cpp -o final -lX11 -lpthread -I. -Isiftpp -O3 siftpp/sift.cpp

clean:
	rm final
