from module_fold.CEModule import define_part_encoder
from module_fold.FMModule import FMModule
from module_fold.ISModule import Generator
from manifold_projection.ManifoldProjection import KNN
import torch


def get_part_vector(part, patch):
    model = define_part_encoder(
        model=part, norm="istance", input_nc=1, latent_dim=512)
    model.load_state_dict(torch.load(pth파일 경로))
    model.eval()
    temp_vector = model(patch)
    return KNN(temp_vector)


def inference(image):
    parts = {"": (0, 0, 512),
             'eye1': (108, 156, 128),
             'eye2': (255, 156, 128),
             'nose': (182, 232, 160),
             'mouth': (169, 301, 192)}

    part_vector = {}

    for part in parts.keys():
        part_vector[part] = get_part_vector(
            part, image[parts[part][1]:parts[part][1]+parts[part][2], parts[part][0]:parts[part][0]+parts[part][2], :])

    Decoder_model = {}
    for part in parts.keys():
        Decoder_model[part] = FMModule(
            norm_layer, image_size=parts[part][2], output_nc=32, latent_dim=512)
        Decoder_model[part].load_state_dict(torch.load())

    whole_feature = Decoder_model[""](part_vector[""])
    whole_feature[parts["eye1"]]

    generator = Generator(input_nc=32, output_nc=3, ngf=56, n_downsampling=3,
                          n_blocks=9)

    generator.load_state_dict(torch.load())
    fakes = generator(whole_feature)
    return fakes


if __name__ == "__main__":
    inference(image)
