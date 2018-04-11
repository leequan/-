QT += core
QT -= gui

TARGET = change_hjl
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += \
    changerIO.cpp \
    test_changerDll.cpp \
    change.cpp \
    fft.cpp \
    matlab.cpp

HEADERS += \
    change.hpp \
    changerIO.hpp \
    changerDll.hpp


INCLUDEPATH += /usr/local/include \
               /usr/local/include/opencv \
                /usr/local/include/opencv2

LIBS += -L/usr/local/lib/ -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_imgproc




