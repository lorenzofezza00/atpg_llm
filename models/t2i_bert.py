import sys
sys.path.append("./DF_GAN/code/")
import pprint
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import torch.nn as nn
from transformers import BertModel, BertConfig
from DF_GAN.code.lib.utils import mkdir_p,get_rank,merge_args_yaml,get_time_stamp,save_args, truncated_noise
from DF_GAN.code.lib.utils import load_model_opt,save_models,load_npz, params_count, load_netG
from DF_GAN.code.lib.perpare import prepare_dataloaders,prepare_models
from DF_GAN.code.lib.modules import sample_one_batch as sample, test as test, train as train, predict_loss, MA_GP, eval as eval, save_single_imgs
from DF_GAN.code.lib.datasets import get_fix_data, prepare_data
import os.path as osp
from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import time
import torchvision.utils as vutils

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))

class Args:
    bert_model='bert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONFIG_NAME = "bird"

    dataset_name = "birds"
    data_dir = "../data/birds"

    gpu_id = 0
    imsize = 256
    z_dim = 100
    cond_dim = 256
    manual_seed = 100
    cuda = True

    stamp = "normal"
    state_epoch = 0
    max_epoch = 1301
    batch_size = 32
    nf = 32
    ch_size = 3

    model = "base"

    gen_interval = 1
    test_interval = 10
    save_interval = 10

    truncation = True
    trunc_rate = 0.88

    sample_times = 10
    npz_path = "../data/birds/npz/bird_val256_FIDK0.npz"
    example_captions = "./example_captions/bird.txt"
    samples_save_dir = "./samples/bird/"
    checkpoint = "./saved_models/bird/pretrained/state_epoch_1220.pth"

    save_image= False
    val_save_dir= "./vals/bird/"

    class TEXT:
        WORDS_NUM= 18
        EMBEDDING_DIM= 256
        CAPTIONS_PER_IMAGE= 10
        DAMSM_NAME= "../data/birds/DAMSMencoder/text_encoder200.pth"
        
    local_rank = 1
    # TODO: Capire cosa metterci
    vocab_size = 10

class TrainArgs(Args):
    imsize=256
    num_workers=4
    batch_size=32
    stamp="normal"
    train=True
    resume_epoch=1
    resume_model_path="./saved_models/bird/base_z_dim100_bird_256_2022_06_04_23_20_33/"
    multi_gpus=True
    nproc_per_node=1
    master_port=11111

class TestArgs(Args):
    batch_size=32
    multi_gpus=True
    master_port=11122
    nproc_per_node=1


class Bert_DFGAN:
    def __init__(self, bert_model='bert-base-uncased', use_large=False):
        super().__init__()
        
        # Scelta tra BERT base o large
        if use_large:
            bert_model = 'bert-large-uncased'
        
        # Encoder BERT
        self.bert = BertModel.from_pretrained(bert_model)
        # Decoder DF-GAN
        image_encoder, _, netG, netD, netC = prepare_models(TestArgs)
        # self.df_gan = {
        #     "img_enc": image_encoder,
        #     # "txt_enc": text_encoder,
        #     "txt_enc": self.bert,
        #     "gan_models": [netG, netD, netC]
        # }
        self.img_enc = image_encoder
        self.netG = netG
        self.netD = netD
        self.netC = netC
        self.z_dim = TestArgs.z_dim
        self.device = TestArgs.device
        self.norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])

    def make_std_inference(self, data, args):
        self.bert.eval()
        imgs, sent_emb, words_embs, keys = prepare_data(data, self.bert)
        sent_emb = sent_emb.to(self.device)
        batch_size = sent_emb.size(0)
        self.netG.eval()
        with torch.no_grad():
            if args.truncation==True:
                noise = truncated_noise(batch_size, self.z_dim, args.trunc_rate)
                noise = torch.tensor(noise, dtype=torch.float).to(self.device)
            else:
                noise = torch.randn(batch_size, self.z_dim).to(self.device)
            fake_imgs = self.netG(noise,sent_emb)
            if args.save_imgs==True:
                save_single_imgs(fake_imgs, args.save_dir, time, args.dl_length, i, batch_size)
            fake = self.norm(fake_imgs)
            pred = self.bert(fake)[0]
        return fake, pred

    # def train_epoch(self, data, optimizerD, optimizerG, device):
    #     # prepare_data
    #     imgs, sent_emb, words_embs, keys = prepare_data(data, self.bert)
    #     imgs = imgs.to(device).requires_grad_()
    #     sent_emb = sent_emb.to(device).requires_grad_()
    #     words_embs = words_embs.to(device).requires_grad_()
    #     # predict real
    #     real_features = self.netD(imgs)
    #     pred_real, errD_real = predict_loss(self.netC, real_features, sent_emb, negtive=False)
    #     mis_features = torch.cat((real_features[1:], real_features[0:1]), dim=0)
    #     _, errD_mis = predict_loss(self.netC, mis_features, sent_emb, negtive=True)
    #     # synthesize fake images
    #     noise = torch.randn(batch_size, self.z_dim).to(device)
    #     fake = self.netG(noise, sent_emb)
    #     fake_features = self.netD(fake.detach())
    #     _, errD_fake = predict_loss(self.netC, fake_features, sent_emb, negtive=True)
    #     # MA-GP
    #     errD_MAGP = MA_GP(imgs, sent_emb, pred_real)
    #     # whole D loss
    #     errD = errD_real + (errD_fake + errD_mis)/2.0 + errD_MAGP
    #     # update D
    #     optimizerD.zero_grad()
    #     errD.backward()
    #     optimizerD.step()
    #     # update G
    #     fake_features = self.netD(fake)
    #     output = self.netC(fake_features, sent_emb)
    #     # sim = MAP(image_encoder, fake, sent_emb).mean()
    #     errG = -output.mean()# - sim
    #     optimizerG.zero_grad()
    #     errG.backward()
    #     optimizerG.step()      
    #     return output
    

def train_model(args):
    time_stamp = get_time_stamp()
    stamp = '_'.join([str(args.model),str(args.stamp),str(args.CONFIG_NAME),str(args.imsize),time_stamp])
    args.model_save_file = osp.join(ROOT_PATH, 'saved_models', str(args.CONFIG_NAME), stamp)
    log_dir = osp.join(ROOT_PATH, 'logs/{0}'.format(osp.join(str(args.CONFIG_NAME), 'train', stamp)))
    args.img_save_dir = osp.join(ROOT_PATH, 'imgs/{0}'.format(osp.join(str(args.CONFIG_NAME), 'train', stamp)))
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        mkdir_p(osp.join(ROOT_PATH, 'logs'))
        mkdir_p(args.model_save_file)
        mkdir_p(args.img_save_dir)
    # prepare TensorBoard
    if (args.multi_gpus==True) and (get_rank() != 0):
        writer = None
    else:
        writer = SummaryWriter(log_dir)
    # prepare dataloader, models, data
    train_dl, valid_dl ,train_ds, valid_ds, sampler = prepare_dataloaders(args)
    args.vocab_size = train_ds.n_words
    # image_encoder, text_encoder, netG, netD, netC = prepare_models(args)
    image_encoder, _, netG, netD, netC = prepare_models(args)
    text_encoder = BertModel.from_pretrained(args.bert_model)
    fixed_img, fixed_sent, fixed_z = get_fix_data(train_dl, valid_dl, text_encoder, args)
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        fixed_grid = make_grid(fixed_img.cpu(), nrow=8, normalize=True)
        writer.add_image('fixed images', fixed_grid, 0)
        img_name = 'z.png'
        img_save_path = osp.join(args.img_save_dir, img_name)
        vutils.save_image(fixed_img.data, img_save_path, nrow=8, normalize=True)
    # prepare optimizer
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    D_params = list(netD.parameters()) + list(netC.parameters())
    optimizerD = torch.optim.Adam(D_params, lr=0.0004, betas=(0.0, 0.9))
    m1, s1 = load_npz(args.npz_path)
    # load from checkpoint
    strat_epoch = 1
    if args.resume_epoch!=1:
        strat_epoch = args.resume_epoch+1
        path = osp.join(args.resume_model_path, 'state_epoch_%03d.pth'%(args.resume_epoch))
        netG, netD, netC, optimizerG, optimizerD = load_model_opt(netG, netD, netC, optimizerG, optimizerD, path, args.multi_gpus)
    # print args
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        pprint.pprint(args)
        arg_save_path = osp.join(log_dir, 'args.yaml')
        save_args(arg_save_path, args)
        print("Start Training")
    # Start training
    test_interval,gen_interval,save_interval = args.test_interval,args.gen_interval,args.save_interval
    #torch.cuda.empty_cache()
    for epoch in range(strat_epoch, args.max_epoch, 1):
        if (args.multi_gpus==True):
            sampler.set_epoch(epoch)
        start_t = time.time()
        # training
        args.current_epoch = epoch
        torch.cuda.empty_cache()
        train(train_dl, netG, netD, netC, text_encoder, optimizerG, optimizerD, args)
        #torch.cuda.empty_cache()
        # save
        if epoch%save_interval==0:
            save_models(netG, netD, netC, optimizerG, optimizerD, epoch, args.multi_gpus, args.model_save_file)
        # sample
        if epoch%gen_interval==0:
            sample(fixed_z, fixed_sent, netG, args.multi_gpus, epoch, args.img_save_dir, writer)
        # end epoch
        # test
        if epoch%test_interval==0:
            torch.cuda.empty_cache()
            fid = test(valid_dl, text_encoder, netG, args.device, m1, s1, epoch, args.max_epoch, \
                        args.sample_times, args.z_dim, args.batch_size, args.truncation, args.trunc_rate)
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            if epoch%test_interval==0:
                writer.add_scalar('FID', fid, epoch)
                print('The %d epoch FID: %.2f'%(epoch,fid))
            end_t = time.time()
            print('The epoch %d costs %.2fs'%(epoch, end_t-start_t))
            print('*'*40)
            
def test_fid(args):
    multi_gpus = args.multi_gpus
    epoch = int(args.checkpoint.split('.')[-2].split('_')[-1])
    time_stamp = get_time_stamp()
    args.val_save_dir = osp.join(args.val_save_dir, time_stamp)
    if args.save_image==True:
        if (multi_gpus==True) and (get_rank() != 0):
            None
        else:
            mkdir_p(args.val_save_dir)
    # prepare data
    train_dl, valid_dl ,train_ds, valid_ds, _ = prepare_dataloaders(args)
    args.vocab_size = train_ds.n_words
    # prepare models
    _, text_encoder, netG, _, _ = prepare_models(args)
    model_path = osp.join(ROOT_PATH, args.checkpoint)
    netG = load_netG(netG, model_path, multi_gpus, train=False)
    netG.eval()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        pprint.pprint(args)
        print('Load %s for NetG'%(args.checkpoint))
        print("************ Start testing FID ************")
    start_t = time.time()
    m1, s1 = load_npz(args.npz_path)
    with torch.no_grad():
        fid = eval(valid_dl, text_encoder, netG, args.device, m1, s1, args.save_image, args.val_save_dir, \
                        args.sample_times, args.z_dim, args.batch_size, args.truncation, args.trunc_rate)
    end_t = time.time()
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        print('Sampling done, %.2fs cost, The FID is : %.2f'%(end_t-start_t, fid))
        
if __name__ == "__main__":
    # Inizializzazione del modello
    # model = Bert_DFGAN(use_large=False)  # Per BERT large: use_large=True
    # model.make_std_inference("Generate an image test library for a multiplier", TestArgs)
    
    # Training Modello
    train_model(TrainArgs)