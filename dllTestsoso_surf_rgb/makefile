
	 
CC=g++
test_changerDll : libchangerIO.so test_changerDll.o 
	$(CC)  test_changerDll.o `pkg-config --libs --cflags opencv` -I. -L. -lchangerIO -o test_changerDll
test_changerDll.o : test_changerDll.cpp
	$(CC) -c  `pkg-config --cflags opencv` test_changerDll.cpp -o test_changerDll.o
libchangerIO.so : changerDll.o  changerIO.o
	$(CC) changerDll.o  changerIO.o -shared -fPIC `pkg-config --libs --cflags opencv` -I. -L. -lsift -o libchangerIO.so	 
changerDll.o :  changerDll.cpp  changerDll.hpp 
	$(CC) -c -fPIC `pkg-config  --cflags opencv` changerDll.cpp  -o changerDll.o
#ftp_client1.o : ftp_client1.cpp  ftp_client.hpp
#	$(CC) -c -fPIC ftp_client1.cpp -o ftp_client1.o
changerIO.o :  changerIO.cpp  changerIO.hpp 
	$(CC) -c -fPIC `pkg-config  --cflags opencv` changerIO.cpp -o changerIO.o 
clean :
	rm -rf *.o *IO.so test_changerDll

#正确
#test_changerDll : libchangerIO.so test_changerDll.cpp
#	g++ test_changerDll.cpp `pkg-config --libs --cflags opencv` -I . -L . -lchangerIO -o test_changerDll
#libchangerIO.so : changerIO.cpp changerDll.cpp ftp_client1.cpp
#	g++ changerIO.cpp changerDll.cpp ftp_client1.cpp -fPIC `pkg-config --cflags --libs opencv` -shared -I . -L . -lsift -o libchangerIO.so
#clean :
#	rm -rf  *.o *IO.so test_changerDll
#g++ changerIO.cpp changerDll.cpp ftp_client.cpp -fPIC `pkg-config --cflags --libs opencv` -shared -I. -L. -lsift -o libchangerIO.so
#g++ test_changerDll.cpp `pkg-config --libs --cflags opencv` -I. -L. -lchangerIO -o test_changerDll


