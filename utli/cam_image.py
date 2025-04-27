""""
Grad-CAM visualization
Support for poolformer, deit, resmlp, resnet, swin and convnext
Modifed from: https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py

please install the following packages
`pip install grad-cam timm`

In the appendix of MetaFormer paper, we use --model=
["poolformer_s24", "resnet50", "deit_small_patch16_224", "resmlp_24_224", "resize"]
for visualization in the appendix. "resize" means resizing the image to resolution 224x224.
The images we shown in the appenix are from ImageNet valdiation set:
val/n02123045/ILSVRC2012_val_00023779.JPEG
val/n03063599/ILSVRC2012_val_00016576.JPEG
val/n01833805/ILSVRC2012_val_00005779.JPEG
val/n07873807/ILSVRC2012_val_00018461.JPEG
"""
import argparse
import os
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from PIL import Image
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import timm
from timm.models import create_model, load_checkpoint
from model import *

def reshape_transform_resmlp(tensor, height=14, width=14):
    result = tensor.reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_swin(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='/datasets/tests/SICE/low_128_HQ/0000001.png',
        help='Input image path')
    parser.add_argument(
        '--output-image-path',
        type=str,
        default='/results/vis/0000001vs.png',
        help='Output image path')
    parser.add_argument(
        '--model',
        type=str,
        default='PINet',

        help='model name')
    parser.add_argument(
        '--pth_path',
        type=str,
        default='/train_weights/PINet/best_psnr_PairLIELoss.pth',
        help='pth_path name') 
    parser.add_argument(
        '--num_classes',
        type=int,
        default=1000,
        help='model name')    
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')                         
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)') 
    parser.add_argument('--bn-tf', action='store_true', default=False,
                        help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='convert model torchscript for inference') 
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)') 
    
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        print(self.features)
        return cos(model_output, self.features)
        
def get_image_from_url(url):
    """A function that gets a URL of an image, 
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model """

    img = np.array(Image.open(url))
    img = cv2.resize(img, (512, 512))
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor
    

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
    if args.model == "HgNet":
        print('===> Building model HgNet')
        model= HgNet()
    elif args.model == "PINet":
        print('===> Building model PINet')
        model= PINet()
    elif args.model == "PINetv2":
        print('===> Building model PINetv2')
        model= PINetv2()
    elif args.model == "PairLIE":
        print('===> Building model PairLIE')
        model= pairlie()
        
    checkpoint = torch.load(args.pth_path)    
    model.load_state_dict(checkpoint, strict=False)

    reshape_transform = None
    
    if 'PINet' in args.model:
        #print(model)
        target_layers = [model.pi.p.downd] 
        print(target_layers)
    elif 'PINetv2' in args.model:
        target_layers = [model.pi[1]]


    model.eval()
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    # target_layers = [model.layer4]
    # import pdb; pdb.set_trace()
    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    
    #img_path = args.image_path
    img, img_float, img_tensor = get_image_from_url(args.image_path)
    concept_features = model(img_tensor)
    targets = [SimilarityToConceptTarget(concept_features)]

    # Where is the car in the image
    with GradCAM(model=model,
                 target_layers=target_layers,
                 use_cuda=False) as cam:
        grayscale_cam = cam(input_tensor=img_tensor,
                            targets=targets)[0, :]
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    #Image.fromarray(cam_image)
    cv2.imwrite(args.output_image_path, cam_image)
    '''
    if args.image_path:
        img_path = args.image_path
    else:
        import requests
        image_url = ''
        img_path = image_url.split('/')[-1]
        if os.path.exists(img_path):
            img_data = requests.get(image_url).content
            with open(img_path, 'wb') as handler:
                handler.write(img_data)
        
    if args.output_image_path:
        save_name = args.output_image_path
    else:
        img_type = img_path.split('.')[-1]
        it_len = len(img_type)
        save_name = img_path.split('/')[-1][:-(it_len + 1)]
        save_name = save_name + '_' + args.model + '.' + img_type

    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    if args.model == 'resize':
        cv2.imwrite(save_name, img)
    else:
        rgb_img = img[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # You can target specific categories by
        # targets = [e.g ClassifierOutputTarget(281)]
        targets = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda,
                        reshape_transform=reshape_transform, 
                        ) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 1
            print(input_tensor)
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        # gb = gb_model(input_tensor, target_category=None)

        # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        # cam_gb = deprocess_image(cam_mask * gb)
        # gb = deprocess_image(gb)

        # cv2.imwrite(f'{args.method}_cam_poolformer_s24.jpg', cam_image)
        
        cv2.imwrite(save_name, cam_image)
        # cv2.imwrite(f'{args.method}_gb_poolformer_s24.jpg', gb)
        # cv2.imwrite(f'{args.method}_cam_gb_poolformer_s24.jpg', cam_gb)
        '''
