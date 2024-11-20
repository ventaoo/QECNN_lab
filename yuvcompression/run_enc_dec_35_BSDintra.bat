x265  --input-res 480x320 --fps 30 --input BSD500train.yuv -o test.hevc --crf 35 --keyint 1 --preset medium
TAppDecoder.exe -b test.hevc -o BSD500trainc.yuv
pause