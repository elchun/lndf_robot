if [ -z $LNDF_SOURCE_DIR ]; then echo 'Please source "lndf_env.sh" first'
else
mkdir $LNDF_SOURCE_DIR/model_weights
wget -O $LNDF_SOURCE_DIR/model_weights/lndf_weights.pth https://www.dropbox.com/s/mtni5sh01dxxjs7/lndf_weights.pth?dl=0
fi