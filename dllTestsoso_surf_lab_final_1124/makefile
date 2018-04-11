
	 
CC=g++
test_changerDll : libchangerIO.so test_changerDll.o 
	$(CC)  test_changerDll.o `pkg-config --libs --cflags opencv` -I. -L. -lchangerIO -o test_changerDll
test_changerDll.o : test_changerDll.cpp
	$(CC) -c  `pkg-config --cflags opencv` test_changerDll.cpp -o test_changerDll.o
libchangerIO.so : changerDll.o  changerIO.o
	$(CC) changerDll.o  changerIO.o -shared -fPIC `pkg-config --libs --cflags opencv` -I. -L. -o libchangerIO.so	 
changerDll.o :  changerDll.cpp  changerDll.hpp 
	$(CC) -c -fPIC `pkg-config  --cflags opencv` changerDll.cpp  -o changerDll.o
changerIO.o :  changerIO.cpp  changerIO.hpp  hammingsse.hpp  brisk.hpp  libopencv_imgproc.so.2.2 libopencv_imgproc.so  libopencv_highgui.so.2.2  libopencv_highgui.so  libopencv_flann.so.2.2  libopencv_flann.so  libopencv_features2d.so.2.2  libopencv_features2d.so  libopencv_core.so.2.2  libopencv_core.so  libopencv_calib3d.so.2.2  libopencv_calib3d.so  libbrisk.so  libagast.so
	$(CC) -c -fPIC -mssse3 `pkg-config  --libs --cflags opencv` -I. -L. -lopencv_imgproc -lopencv_highgui -lopencv_flann -lopencv_features2d -lopencv_core -lopencv_calib3d -lbrisk -lagast changerIO.cpp -o changerIO.o 
clean :
	rm -rf *.o *IO.so test_changerDll



