QT += core
QT -= gui

TARGET = change_zt
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += \
    changerDll.cpp \
    changerIO.cpp \
    test_changerDll.cpp

HEADERS += \
    changerDll.hpp \
    changerIO.hpp

INCLUDEPATH += /usr/local/include \
               /usr/local/include/opencv \
                /usr/local/include/opencv2

LIBS += -L/usr/local/lib/ -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_nonfree -lopencv_features2d \
                  -lopencv_objdetect -lopencv_legacy  -lopencv_flann -lopencv_calib3d
