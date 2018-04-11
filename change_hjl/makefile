
	 
CC=g++
test_changerDll : libchangerIO.so test_changerDll.o 
	$(CC)  test_changerDll.o `pkg-config --libs --cflags opencv` -I. -L. -lchangerIO -o test_changerDll
test_changerDll.o : test_changerDll.cpp
	$(CC) -c  `pkg-config --cflags opencv` test_changerDll.cpp -o test_changerDll.o
libchangerIO.so : changerIO.o  change.o fft.o  matlab.o
	$(CC) change.o fft.o  matlab.o  changerIO.o -shared -fPIC `pkg-config --libs --cflags opencv` -I. -L. -o libchangerIO.so	 
changerIO.o :  changerIO.cpp  changerIO.hpp 
	$(CC) -c -fPIC `pkg-config  --cflags opencv` changerIO.cpp -o changerIO.o 
change.o :  change.cpp  change.hpp 
	$(CC) -c -fPIC `pkg-config  --cflags opencv` change.cpp -o change.o 
fft.o :  fft.cpp  
	$(CC) -c -fPIC `pkg-config  --cflags opencv` fft.cpp -o fft.o 
matlab.o :  matlab.cpp  
	$(CC) -c -fPIC `pkg-config  --cflags opencv` matlab.cpp -o matlab.o 
clean :
	rm -rf *.o *IO.so test_changerDll



