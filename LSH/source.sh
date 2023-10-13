now=$(date +"%y%m%d.%H%M%S")
filename="logs/log_$now.log"
nohup python glove.py > "$filename" &

