QT += core
QT -= gui

TARGET = change_htj
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += \
    change.cpp \
    fft.cpp \
    matlab.cpp

HEADERS += \
    change.h

INCLUDEPATH += /usr/local/include \
               /usr/local/include/opencv \
                /usr/local/include/opencv2

LIBS += -L/usr/local/lib/ -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_nonfree -lopencv_features2d \
                  -lopencv_objdetect -lopencv_legacy  -lopencv_flann -lopencv_calib3d
