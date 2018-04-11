#for((i=1;i<10;i++));

#do

#echo $i;

#done
#”
#!/bin/bash

VARR1=" "
VARR2=" " 
VARR3=" " 
VARR4="ENDL"
VARRTEMP=" "
while read line
do
 
  VARR1=${line%%&*}
  VARRTEMP=${line%&*}
  VARR2=${VARRTEMP##*&}
  VARR3=${line##*&}
  #echo $VARR1
  #echo $VARR2
  #echo $VARR3
  #echo $VARR4
  ./test_changerDll $VARR1 $VARR2 $VARR3

done < "imageslist1.txt"
#(待读取的文件)
