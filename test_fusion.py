import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from model.IFFusion_main import IFFusion
from torch.utils.data import DataLoader
from tqdm import tqdm
from create_dataset import Fusion_Data
from utils import *
import torchvision

# Training settings
parser = argparse.ArgumentParser(description='IFFusion')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--data_test', type=str, default='./dataset')
parser.add_argument('--data_name', type=str, default='MSRS')
parser.add_argument('--model', default='weights/IFFusion.pth', help='Pretrained base model')
parser.add_argument('--output_folder', type=str, default='./fused_img')
parser.add_argument('--stage', type=str, default='test')
opt = parser.parse_args()

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
def test():
    torch.set_grad_enabled(False)
    fusion_model.eval()
    with torch.no_grad():
        with tqdm(testing_data_loader, desc='IFFusion') as pbar:
            for iteration, batch in enumerate(testing_data_loader, 1):
                img_vi, img_ir, name = batch[0], batch[1], batch[2]
                img_ir = img_ir.cuda()
                img_vi = img_vi.cuda()
                R1, R_fuse, L1, weights = fusion_model(img_ir, img_vi)
                L1 = torch.max(L1, img_ir[:, 0:1])
                fused_img = torch.pow(L1, weights*0.2) * R_fuse
                save_img = os.path.join(opt.output_folder, opt.data_name)
                os.makedirs(save_img, exist_ok=True)
                save_path = os.path.join(save_img, name[0])
                torchvision.utils.save_image(fused_img, save_path, nrow=1)
                pbar.set_postfix_str(f"{name[0]} has saved successfully")
                pbar.update(1)
def checkpoint(epoch):
    model_out_path = opt.save_folder+"epoch_{}.pth".format(epoch)
    torch.save(fusion_model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
print('===> Building model ')
fusion_model = IFFusion().cuda()
fusion_model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('===> Loading datasets')

test_data_path = os.path.join(opt.data_test, opt.data_name)
test_set = Fusion_Data(test_data_path, is_train=False)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

if not os.path.exists(opt.output_folder):
    os.mkdir(opt.output_folder)

if __name__ == "__main__":
    test()