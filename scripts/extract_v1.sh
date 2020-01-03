line=$1
mkdir ${line%????}
name1=$(echo $1 |sed "s/.*\///")
name=${name1%????}
echo $name
FFREPORT=file=${line%????}/info.txt:level=32 ffmpeg -ss 00:00:00 -i ${line} -vf '[in]select=eq(pict_type\,I),showinfo[out]' -vsync 0 ${line%????}/$name%03d.jpg
cat	${line%????}/info.txt  | grep -o 'pts_time:[^ ,]\+' > ${line%????}/$name.txt
cat     ${line%????}/info.txt  | grep -o -m 1 '..... fps' >> ${line%????}/$name.txt
rm ${line%????}/info.txt
#run parallely after installing sudo apt install parallel 
# ls -R | grep mp4 | parallel -j 20 /home/teh_devs/deepfake/deepfake-detection/extract_v1.sh