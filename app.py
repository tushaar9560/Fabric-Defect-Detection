import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
from skimage import segmentation
import torch.nn.init
from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename

os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = Flask(__name__, template_folder='template')

use_cuda = torch.cuda.is_available()


args = {
    'nChannel': 100,
    'lr': 0.1,
    'maxIter': 10,
    'minLabels': 2,
    'compactness': 100,
    'nConv': 2,
    'visualize': 100,
    'num_superpixels': 10000,    
}

class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args['nChannel'], kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args['nChannel'])
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args['nConv']-1):
            self.conv2.append( nn.Conv2d(args['nChannel'], args['nChannel'], kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args['nChannel']) )
        self.conv3 = nn.Conv2d(args['nChannel'], args['nChannel'], kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args['nChannel'])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args['nConv']-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x



@app.route("/", methods=["GET","POST"])

def index():
    result_image_path = None
    if request.method=='POST':

        file = request.files['file']
        if file and allowed_file(file.filename):
            image_path = os.path.join('inputs', secure_filename(file.filename))
            file.save(image_path)

            result_image_path = detect_defect(image_path)
    
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg','jpeg','png','gif'}


def detect_defect(image_path):
    im = cv2.imread(image_path)
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    # slic
    labels = segmentation.slic(im, compactness=args['compactness'], n_segments=args['num_superpixels'])
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

    # train
    model = MyNet(data.size(1))
    if use_cuda:
        model.cuda()
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)
    label_colours = np.random.randint(255,size=(100,3))
    for batch_idx in range(args['maxIter']):
        # forwarding
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1,args['nChannel'] )
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
        filename = os.path.basename(image_path)
        # To visualize
        if args['visualize']:
            cv2.imshow( filename, im_target_rgb )
            cv2.waitKey(10)

        # superpixel refinement
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
            im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
        target = torch.from_numpy( im_target )

        if use_cuda:
            target = target.cuda()
        target = Variable( target )
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


        # print (batch_idx, '/', args['maxIter'], ':', nLabels, loss.item())

        if nLabels <= args['minLabels']:
            # print ("nLabels", nLabels, "reached minLabels", args['minLabels'], ".")
            break

    result_image_path = os.path.join('outputs', filename)

    cv2.imwrite(result_image_path, im_target_rgb )
    print(result_image_path)
    return result_image_path

if __name__ == "__main__":
    app.run('0.0.0.0',debug=False)

