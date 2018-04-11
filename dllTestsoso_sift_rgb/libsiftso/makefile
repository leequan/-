

CC=gcc
SOUCE_FILE=  imgfeatures.c kdtree.c  minpq.c  sift.c utils.c xform.c
OBJECT_FILE=  imgfeatures.o kdtree.o minpq.o sift.o utils.o xform.o

CFLAGS= `pkg-config --cflags opencv`
LIBS=`pkg-config --libs opencv`

#main : libsift.so main.o 
#	g++ -o main main.o  $(LIBS)  -L.  -lsift 
#main.o : main.c
#	g++ -c  $(CFLAGS)  main.c -o main.o 
libsift.so : $(OBJECT_FILE) 
	$(CC) -shared -fPIC -o libsift.so $(OBJECT_FILE)
#mymosaic.o : mymosaic.c
#	g++ -c -fPIC $(CFLAGS) mymosaic.c -o mymosaic.o
$(OBJECT_FILE) : $(SOUCE_FILE) 
	$(CC) -c -fPIC $(CFLAGS) $(SOUCE_FILE)
clean :
	@-rm -rf *.so main
	





