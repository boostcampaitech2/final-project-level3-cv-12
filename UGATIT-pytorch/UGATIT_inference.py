import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
import wandb
import ADA
class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset


        self.batch_size = args.batch_size

        self.ch = args.ch

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        
        self.iter_for_inference = args.iter_for_inference

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True



    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """

        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])


        self.sketch_for_inference = ImageFolder(os.path.join('dataset', self.dataset, 'sketch_for_inference'), test_transform)
        self.inference_loader = DataLoader(self.sketch_for_inference, batch_size=1, shuffle=False)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)



    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])



    
    def inference(self):

        iter = 92000
        self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
        print(" [*] Load SUCCESS")


        self.genA2B.eval()
        for n, (real_A, _) in enumerate(self.inference_loader):
            print("###")
            real_A = real_A.to(self.device)

            fake_A2B, _, _ = self.genA2B(real_A)

            A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),), 0)
            
            inference_dir = os.path.join(self.result_dir, self.dataset, 'inference', str(iter))
            if not os.path.exists(inference_dir):
                os.makedirs(inference_dir)

            cv2.imwrite(os.path.join(inference_dir, 'A2B_%d.png' % (n + 1)), A2B * 255.0)
