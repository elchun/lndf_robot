trap "kill $LASTPID" INT

CUDA_VISIBLE_DEVICES=0 python3 reconstruction_test.py
python3 -m http.server &
LASTPID=$!

sleep 10m; kill $LASTPID
echo "Server closed"
