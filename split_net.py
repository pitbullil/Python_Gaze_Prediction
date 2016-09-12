s3cnn_weights = {}
for key in anet_weights.keys():
    if key != 'fc8':
        s3cnn_weights[('sp_'+key)]= anet_weights[key]
        s3cnn_weights[('nn_'+key)]= anet_weights[key]
        s3cnn_weights[('pic_'+key)]= anet_weights[key]
s3cnn_weights['nn1']=s3cnn['nn1']
s3cnn_weights['nn2']=s3cnn['nn2']
s3cnn_weights['nout']=s3cnn['nout']