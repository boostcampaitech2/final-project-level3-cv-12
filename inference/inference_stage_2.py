from module_fold.CEModule import define_part_encoder, define_part_decoder
from module_fold.ISModule import Generator
from manifold_projection.ManifoldProjection import KNN
import torch
import os
import numpy as np
import random
import torch.nn as nn


def inference_patch_img(patch_dict, encoder_pth_path, decoder_pth_path):
    after_knn_patch = {}
    for part in patch_dict.keys():
        encoder_model = define_part_encoder(model=part)
        check_point_encoder = torch.load(
            os.path.join(encoder_pth_path, part+".pth"))
        state_dict_encoder = check_point_encoder.state_dict()
        encoder_model.load_state_dict(state_dict_encoder)
        part_vector = encoder_model(patch_dict[part])
        '''
        knn
        '''
        decoder_model = define_part_decoder(model=part)
        check_point_decoder = torch.load(
            os.path.join(decoder_pth_path, part+".pth"))
        state_dict_decoder = check_point_decoder.state_dict()
        decoder_model.load_state_dict(state_dict_decoder)
        after_knn_patch[part] = decoder_model(part_vector)

    return after_knn_patch


def get_patch_img(image):
    patch = {}
    parts = {'left_eye': (108, 156, 128),
             'right_eye2': (255, 156, 128),
             'nose': (182, 232, 160),
             'mouth': (169, 301, 192)}

    for part in parts.keys():
        patch[part] = image[parts[part][1]:parts[part][1]+parts[part]
                            [2], parts[part][0]:parts[part][0]+parts[part][2], :]

    for part in parts.keys():
        image[parts[part][1]:parts[part][1]+parts[part]
              [2], parts[part][0]:parts[part][0]+parts[part][2], :] = 0

    patch["remainer"] = image
    return patch


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def inference(image, generator):
    #--- settings
    seed_everything(42)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # --- get patch img & infer
    patch_dict = get_patch_img(image)
    inferred_patch_dict = inference_patch_img(
        patch_dict, encoder_pth_path=None, decoder_pth_path=None)

    whole_feature = torch.FloatTensor(np.zeros((1, 5, 512, 512))).to(device)
    whole_feature[:, 0:1, :, :] = inferred_patch_dict["remainder"]
    whole_feature[:, 1:2, :, :] = inferred_patch_dict["mouth"]
    whole_feature[:, 2:3, :, :] = inferred_patch_dict["nose"]
    whole_feature[:, 3:4, :, :] = inferred_patch_dict["left_eye"]
    whole_feature[:, 4:5, :, :] = inferred_patch_dict["right_eye"]

    fakes = generator(whole_feature)
    return fakes


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    generator = Generator(input_nc=5, output_nc=3, ngf=56, n_downsampling=3,
                          n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect')
    generator.to(device)
    checkpoint_generator = torch.load(os.path.join(), "generator.pth")
    statedict_generator = checkpoint_generator.state_dict()
    checkpoint_generator.load_state_dict(checkpoint_generator)
    inference(image, generator)
