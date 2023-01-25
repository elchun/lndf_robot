if [ -z $LNDF_SOURCE_DIR ]; then echo 'Please source "lndf_env.sh" first'
else
mkdir $LNDF_SOURCE_DIR/model_weights
wget -O $LNDF_SOURCE_DIR/model_weights/lndf_weights.pth https://www.dropbox.com/s/mtni5sh01dxxjs7/lndf_weights.pth?dl=0
wget -O $LNDF_SOURCE_DIR/model_weights/lndf_no_se3_weights.pth https://www.dropbox.com/s/mqb28hxo0m2r2a5/lndf_no_se3_weights?dl=0
# wget -O $LNDF_SOURCE_DIR/model_weights/lndf_weights_2.pth https://www.dropbox.com/s/kevps1c5a081ib5/lndf_weights_2.pth?dl=0
wget -O $LNDF_SOURCE_DIR/model_weights/ndf_weights.pth https://www.dropbox.com/s/hm4hty56ldu1wb5/multi_category_weights.pth?dl=0
fi