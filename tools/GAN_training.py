'''
GAN filter for detection to reduce jitters in the detection / conversion of pose to 3d Points
'''
import torch
import random
from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.utils.data import DataLoader
import tensorflow as tf
import time
from torch.utils.data._utils.collate import default_collate
from os.path import join as pjoin
import os
import numpy as np
import math

def print_current_loss(start_time, niter_state, losses, epoch=None, sub_epoch=None,
                       inner_iter=None, tf_ratio=None, sl_steps=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None:
        print('epoch: %3d niter: %6d sub_epoch: %2d inner_iter: %4d' % (epoch, niter_state, sub_epoch, inner_iter), end=" ")

    # message = '%s niter: %d completed: %3d%%)' % (time_since(start_time, niter_state / total_niters),
    #                                             niter_state, niter_state / total_niters * 100)
    now = time.time()
    message = '%s'%(as_minutes(now - start_time))

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    message += ' sl_length:%2d tf_ratio:%.2f'%(sl_steps, tf_ratio)
    print(message)

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

class Logger(object):
  def __init__(self, log_dir):
    self.writer = tf.summary.create_file_writer(log_dir)

  def scalar_summary(self, tag, value, step):
      with self.writer.as_default():
          tf.summary.scalar(tag, value, step=step)
          self.writer.flush()

class GANTrainer(object):

    def __init__(self, args, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=None, seq_post=None):
        self.opt = args
        self.text_enc = text_enc
        self.seq_pri = seq_pri
        self.att_layer = att_layer
        self.device = args.device
        self.seq_dec = seq_dec
        self.mov_dec = mov_dec
        self.mov_enc = mov_enc

        if args.is_train:
            self.seq_post = seq_post
            # self.motion_dis
            self.logger = Logger(args.log_dir)
            self.l1_criterion = torch.nn.SmoothL1Loss()
            self.gan_criterion = torch.nn.BCEWithLogitsLoss()
            self.mse_criterion = torch.nn.MSELoss()

    @staticmethod
    def reparametrize(mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu)

    @staticmethod
    def ones_like(tensor, val=1.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(tensor.device).requires_grad_(False)

    @staticmethod
    def zeros_like(tensor, val=0.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(tensor.device).requires_grad_(False)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    @staticmethod
    def kl_criterion(mu1, logvar1, mu2, logvar2):
        # KL( N(mu1, sigma2_1) || N(mu_2, sigma2_2))
        # loss = log(sigma2/sigma1) + (sigma1^2 + (mu1 - mu2)^2)/(2*sigma2^2) - 1/2
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (
                2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / mu1.shape[0]

    @staticmethod
    def kl_criterion_unit(mu, logvar):
        # KL( N(mu1, sigma2_1) || N(mu_2, sigma2_2))
        # loss = log(sigma2/sigma1) + (sigma1^2 + (mu1 - mu2)^2)/(2*sigma2^2) - 1/2
        kld = ((torch.exp(logvar) + mu ** 2)  - logvar  - 1) / 2
        return kld.sum() / mu.shape[0]

    def forward(self, batch_data, tf_ratio, mov_len, eval_mode=False):
        word_emb, pos_ohot, caption, cap_lens, motions, m_lens = batch_data
        word_emb = word_emb.detach().to(self.device).float()
        pos_ohot = pos_ohot.detach().to(self.device).float()
        motions = motions.detach().to(self.device).float()
        self.cap_lens = cap_lens
        self.caption = caption

        # print(motions.shape)
        # (batch_size, motion_len, pose_dim)
        self.motions = motions

        '''Movement Encoding'''
        self.movements = self.mov_enc(self.motions[..., :-4]).detach()
        # Initially input a mean vector
        mov_in = self.mov_enc(
            torch.zeros((self.motions.shape[0], self.opt.unit_length, self.motions.shape[-1] - 4), device=self.device)
        ).squeeze(1).detach()
        assert self.movements.shape[1] == mov_len

        teacher_force = True if random.random() < tf_ratio else False

        '''Text Encoding'''
        # time0 = time.time()
        # text_input = torch.cat([word_emb, pos_ohot], dim=-1)
        word_hids, hidden = self.text_enc(word_emb, pos_ohot, cap_lens)
        # print(word_hids.shape, hidden.shape)

        if self.opt.text_enc_mod == 'bigru':
            hidden_pos = self.seq_post.get_init_hidden(hidden)
            hidden_pri = self.seq_pri.get_init_hidden(hidden)
            hidden_dec = self.seq_dec.get_init_hidden(hidden)
        elif self.opt.text_enc_mod == 'transformer':
            hidden_pos = self.seq_post.get_init_hidden(hidden.detach())
            hidden_pri = self.seq_pri.get_init_hidden(hidden.detach())
            hidden_dec = self.seq_dec.get_init_hidden(hidden)

        mus_pri = []
        logvars_pri = []
        mus_post = []
        logvars_post = []
        fake_mov_batch = []

        query_input = []

        # time1 = time.time()
        # print("\t Text Encoder Cost:%5f" % (time1 - time0))
        # print(self.movements.shape)

        for i in range(mov_len):
            # print("\t Sequence Measure")
            # print(mov_in.shape)
            mov_tgt = self.movements[:, i]
            '''Local Attention Vector'''
            att_vec, _ = self.att_layer(hidden_dec[-1], word_hids)
            query_input.append(hidden_dec[-1])

            tta = m_lens // self.opt.unit_length - i

            if self.opt.text_enc_mod == 'bigru':
                pos_in = torch.cat([mov_in, mov_tgt, att_vec], dim=-1)
                pri_in = torch.cat([mov_in, att_vec], dim=-1)

            elif self.opt.text_enc_mod == 'transformer':
                pos_in = torch.cat([mov_in, mov_tgt, att_vec.detach()], dim=-1)
                pri_in = torch.cat([mov_in, att_vec.detach()], dim=-1)

            '''Posterior'''
            z_pos, mu_pos, logvar_pos, hidden_pos = self.seq_post(pos_in, hidden_pos, tta)

            '''Prior'''
            z_pri, mu_pri, logvar_pri, hidden_pri = self.seq_pri(pri_in, hidden_pri, tta)

            '''Decoder'''
            if eval_mode:
                dec_in = torch.cat([mov_in, att_vec, z_pri], dim=-1)
            else:
                dec_in = torch.cat([mov_in, att_vec, z_pos], dim=-1)
            fake_mov, hidden_dec = self.seq_dec(dec_in, mov_in, hidden_dec, tta)

            # print(fake_mov.shape)

            mus_post.append(mu_pos)
            logvars_post.append(logvar_pos)
            mus_pri.append(mu_pri)
            logvars_pri.append(logvar_pri)
            fake_mov_batch.append(fake_mov.unsqueeze(1))

            if teacher_force:
                mov_in = self.movements[:, i].detach()
            else:
                mov_in = fake_mov.detach()


        self.fake_movements = torch.cat(fake_mov_batch, dim=1)

        # print(self.fake_movements.shape)

        self.fake_motions = self.mov_dec(self.fake_movements)

        self.mus_post = torch.cat(mus_post, dim=0)
        self.mus_pri = torch.cat(mus_pri, dim=0)
        self.logvars_post = torch.cat(logvars_post, dim=0)
        self.logvars_pri = torch.cat(logvars_pri, dim=0)

    def generate(self, word_emb, pos_ohot, cap_lens, m_lens, mov_len, dim_pose):
        word_emb = word_emb.detach().to(self.device).float()
        pos_ohot = pos_ohot.detach().to(self.device).float()
        self.cap_lens = cap_lens

        # print(motions.shape)
        # (batch_size, motion_len, pose_dim)

        '''Movement Encoding'''
        # Initially input a mean vector
        mov_in = self.mov_enc(
            torch.zeros((word_emb.shape[0], self.opt.unit_length, dim_pose - 4), device=self.device)
        ).squeeze(1).detach()

        '''Text Encoding'''
        # time0 = time.time()
        # text_input = torch.cat([word_emb, pos_ohot], dim=-1)
        word_hids, hidden = self.text_enc(word_emb, pos_ohot, cap_lens)
        # print(word_hids.shape, hidden.shape)

        hidden_pri = self.seq_pri.get_init_hidden(hidden)
        hidden_dec = self.seq_dec.get_init_hidden(hidden)

        mus_pri = []
        logvars_pri = []
        fake_mov_batch = []
        att_wgt = []

        # time1 = time.time()
        # print("\t Text Encoder Cost:%5f" % (time1 - time0))
        # print(self.movements.shape)

        for i in range(mov_len):
            # print("\t Sequence Measure")
            # print(mov_in.shape)
            '''Local Attention Vector'''
            att_vec, co_weights = self.att_layer(hidden_dec[-1], word_hids)

            tta = m_lens // self.opt.unit_length - i
            # tta = m_lens - i

            '''Prior'''
            pri_in = torch.cat([mov_in, att_vec], dim=-1)
            z_pri, mu_pri, logvar_pri, hidden_pri = self.seq_pri(pri_in, hidden_pri, tta)

            '''Decoder'''
            dec_in = torch.cat([mov_in, att_vec, z_pri], dim=-1)
            
            fake_mov, hidden_dec = self.seq_dec(dec_in, mov_in, hidden_dec, tta)

            # print(fake_mov.shape)
            mus_pri.append(mu_pri)
            logvars_pri.append(logvar_pri)
            fake_mov_batch.append(fake_mov.unsqueeze(1))
            att_wgt.append(co_weights)

            mov_in = fake_mov.detach()

        fake_movements = torch.cat(fake_mov_batch, dim=1)
        att_wgts = torch.cat(att_wgt, dim=-1)

        # print(self.fake_movements.shape)

        fake_motions = self.mov_dec(fake_movements)

        mus_pri = torch.cat(mus_pri, dim=0)
        logvars_pri = torch.cat(logvars_pri, dim=0)

        return fake_motions, mus_pri, att_wgts

    def backward_G(self):
        self.loss_mot_rec = self.l1_criterion(self.fake_motions, self.motions)
        self.loss_mov_rec = self.l1_criterion(self.fake_movements, self.movements)

        self.loss_kld = self.kl_criterion(self.mus_post, self.logvars_post, self.mus_pri, self.logvars_pri)

        self.loss_gen = self.loss_mot_rec * self.opt.lambda_rec_mov + self.loss_mov_rec * self.opt.lambda_rec_mot + \
                        self.loss_kld * self.opt.lambda_kld
        loss_logs = OrderedDict({})
        loss_logs['loss_gen'] = self.loss_gen.item()
        loss_logs['loss_mot_rec'] = self.loss_mot_rec.item()
        loss_logs['loss_mov_rec'] = self.loss_mov_rec.item()
        loss_logs['loss_kld'] = self.loss_kld.item()

        return loss_logs
        # self.loss_gen = self.loss_rec_mov

        # self.loss_gen = self.loss_rec_mov * self.opt.lambda_rec_mov + self.loss_rec_mot + \
        #                 self.loss_kld * self.opt.lambda_kld + \
        #                 self.loss_mtgan_G * self.opt.lambda_gan_mt + self.loss_mvgan_G * self.opt.lambda_gan_mv


    def update(self):

        self.zero_grad([self.opt_text_enc, self.opt_seq_dec, self.opt_seq_post,
                        self.opt_seq_pri, self.opt_att_layer, self.opt_mov_dec])
        # time2_0 = time.time()
        # print("\t\t Zero Grad:%5f" % (time2_0 - time1))
        loss_logs = self.backward_G()

        # time2_1 = time.time()
        # print("\t\t Backward_G :%5f" % (time2_1 - time2_0))
        self.loss_gen.backward()

        # time2_2 = time.time()
        # print("\t\t Backward :%5f" % (time2_2 - time2_1))
        self.clip_norm([self.text_enc, self.seq_dec, self.seq_post, self.seq_pri,
                        self.att_layer, self.mov_dec])

        # time2_3 = time.time()
        # print("\t\t Clip Norm :%5f" % (time2_3 - time2_2))
        self.step([self.opt_text_enc, self.opt_seq_dec, self.opt_seq_post,
                        self.opt_seq_pri, self.opt_att_layer, self.opt_mov_dec])

        # time2_4 = time.time()
        # print("\t\t Step :%5f" % (time2_4 - time2_3))

        # time2 = time.time()
        # print("\t Update Generator Cost:%5f" % (time2 - time1))

        # self.zero_grad([self.opt_att_layer])
        # self.backward_Att()
        # self.loss_lgan_G_.backward()
        # self.clip_norm([self.att_layer])
        # self.step([self.opt_att_layer])
        # # time3 = time.time()
        # # print("\t Update Att Cost:%5f" % (time3 - time2))

        # self.loss_gen += self.loss_lgan_G_

        return loss_logs

    def to(self, device):
        if self.opt.is_train:
            self.gan_criterion.to(device)
            self.mse_criterion.to(device)
            self.l1_criterion.to(device)
            self.seq_post.to(device)
        self.mov_enc.to(device)
        self.text_enc.to(device)
        self.mov_dec.to(device)
        self.seq_pri.to(device)
        self.att_layer.to(device)
        self.seq_dec.to(device)

    def train_mode(self):
        if self.opt.is_train:
            self.seq_post.train()
        self.mov_enc.eval()
            # self.motion_dis.train()
            # self.movement_dis.train()
        self.mov_dec.train()
        self.text_enc.train()
        self.seq_pri.train()
        self.att_layer.train()
        self.seq_dec.train()


    def eval_mode(self):
        if self.opt.is_train:
            self.seq_post.eval()
        self.mov_enc.eval()
            # self.motion_dis.train()
            # self.movement_dis.train()
        self.mov_dec.eval()
        self.text_enc.eval()
        self.seq_pri.eval()
        self.att_layer.eval()
        self.seq_dec.eval()


    def save(self, file_name, ep, total_it, sub_ep, sl_len):
        state = {
            # 'latent_dis': self.latent_dis.state_dict(),
            # 'motion_dis': self.motion_dis.state_dict(),
            'text_enc': self.text_enc.state_dict(),
            'seq_post': self.seq_post.state_dict(),
            'att_layer': self.att_layer.state_dict(),
            'seq_dec': self.seq_dec.state_dict(),
            'seq_pri': self.seq_pri.state_dict(),
            'mov_enc': self.mov_enc.state_dict(),
            'mov_dec': self.mov_dec.state_dict(),

            # 'opt_motion_dis': self.opt_motion_dis.state_dict(),
            'opt_mov_dec': self.opt_mov_dec.state_dict(),
            'opt_text_enc': self.opt_text_enc.state_dict(),
            'opt_seq_pri': self.opt_seq_pri.state_dict(),
            'opt_att_layer': self.opt_att_layer.state_dict(),
            'opt_seq_post': self.opt_seq_post.state_dict(),
            'opt_seq_dec': self.opt_seq_dec.state_dict(),
            # 'opt_movement_dis': self.opt_movement_dis.state_dict(),

            'ep': ep,
            'total_it': total_it,
            'sub_ep': sub_ep,
            'sl_len': sl_len
        }
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        if self.opt.is_train:
            self.seq_post.load_state_dict(checkpoint['seq_post'])
            # self.opt_latent_dis.load_state_dict(checkpoint['opt_latent_dis'])

            self.opt_text_enc.load_state_dict(checkpoint['opt_text_enc'])
            self.opt_seq_post.load_state_dict(checkpoint['opt_seq_post'])
            self.opt_att_layer.load_state_dict(checkpoint['opt_att_layer'])
            self.opt_seq_pri.load_state_dict(checkpoint['opt_seq_pri'])
            self.opt_seq_dec.load_state_dict(checkpoint['opt_seq_dec'])
            self.opt_mov_dec.load_state_dict(checkpoint['opt_mov_dec'])

        self.text_enc.load_state_dict(checkpoint['text_enc'])
        self.mov_dec.load_state_dict(checkpoint['mov_dec'])
        self.seq_pri.load_state_dict(checkpoint['seq_pri'])
        self.att_layer.load_state_dict(checkpoint['att_layer'])
        self.seq_dec.load_state_dict(checkpoint['seq_dec'])
        self.mov_enc.load_state_dict(checkpoint['mov_enc'])

        return checkpoint['ep'], checkpoint['total_it'], checkpoint['sub_ep'], checkpoint['sl_len']

    def train(self, train_dataset, val_dataset, plot_eval):
        self.to(self.device)

        self.opt_text_enc = optim.Adam(self.text_enc.parameters(), lr=self.opt.lr)
        self.opt_seq_post = optim.Adam(self.seq_post.parameters(), lr=self.opt.lr)
        self.opt_seq_pri = optim.Adam(self.seq_pri.parameters(), lr=self.opt.lr)
        self.opt_att_layer = optim.Adam(self.att_layer.parameters(), lr=self.opt.lr)
        self.opt_seq_dec = optim.Adam(self.seq_dec.parameters(), lr=self.opt.lr)

        self.opt_mov_dec = optim.Adam(self.mov_dec.parameters(), lr=self.opt.lr*0.1)

        epoch = 0
        it = 0
        if self.opt.dataset_name == 't2m':
            schedule_len = 10
        elif self.opt.dataset_name == 'kit':
            schedule_len = 6
        sub_ep = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it, sub_ep, schedule_len = self.load(model_dir)

        invalid = True
        start_time = time.time()
        val_loss = 0
        is_continue_and_first = self.opt.is_continue
        while invalid:
            train_dataset.reset_max_len(schedule_len * self.opt.unit_length)
            val_dataset.reset_max_len(schedule_len * self.opt.unit_length)

            train_loader = DataLoader(train_dataset, batch_size=self.opt.batch_size, drop_last=True, num_workers=4,
                                      shuffle=True, collate_fn=collate_fn, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=self.opt.batch_size, drop_last=True, num_workers=4,
                                      shuffle=True, collate_fn=collate_fn, pin_memory=True)
            print("Max_Length:%03d Training Split:%05d Validation Split:%04d" % (schedule_len, len(train_loader), len(val_loader)))

            min_val_loss = np.inf
            stop_cnt = 0
            logs = OrderedDict()
            for sub_epoch in range(sub_ep, self.opt.max_sub_epoch):
                self.train_mode()

                if is_continue_and_first:
                    sub_ep = 0
                    is_continue_and_first = False

                tf_ratio = self.opt.tf_ratio

                time1 = time.time()
                for i, batch_data in enumerate(train_loader):
                    time2 = time.time()
                    self.forward(batch_data, tf_ratio, schedule_len)
                    time3 = time.time()
                    log_dict = self.update()
                    for k, v in log_dict.items():
                        if k not in logs:
                            logs[k] = v
                        else:
                            logs[k] += v
                    time4 = time.time()


                    it += 1
                    if it % self.opt.log_every == 0:
                        mean_loss = OrderedDict({'val_loss': val_loss})
                        self.logger.scalar_summary('val_loss', val_loss, it)
                        self.logger.scalar_summary('scheduled_length', schedule_len, it)

                        for tag, value in logs.items():
                            self.logger.scalar_summary(tag, value/self.opt.log_every, it)
                            mean_loss[tag] = value / self.opt.log_every
                        logs = OrderedDict()
                        print_current_loss(start_time, it, mean_loss, epoch, sub_epoch=sub_epoch, inner_iter=i,
                                           tf_ratio=tf_ratio, sl_steps=schedule_len)

                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it, sub_epoch, schedule_len)

                    time5 = time.time()
                    # print("Data Loader Time: %5f s" % ((time2 - time1)))
                    # print("Forward Time: %5f s" % ((time3 - time2)))
                    # print("Update Time: %5f s" % ((time4 - time3)))
                    # print('Per Iteration: %5f s' % ((time5 -  time1)))
                    time1 = time5

                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it, sub_epoch, schedule_len)

                epoch += 1
                if epoch % self.opt.save_every_e == 0:
                    self.save(pjoin(self.opt.model_dir, 'E%03d_SE%02d_SL%02d.tar'%(epoch, sub_epoch, schedule_len)),
                              epoch, total_it=it, sub_ep=sub_epoch, sl_len=schedule_len)

                print('Validation time:')

                loss_mot_rec = 0
                loss_mov_rec = 0
                loss_kld = 0
                val_loss = 0
                with torch.no_grad():
                    for i, batch_data in enumerate(val_loader):
                        self.forward(batch_data, 0, schedule_len)
                        self.backward_G()
                        loss_mot_rec += self.loss_mot_rec.item()
                        loss_mov_rec += self.loss_mov_rec.item()
                        loss_kld += self.loss_kld.item()
                        val_loss += self.loss_gen.item()

                loss_mot_rec /= len(val_loader) + 1
                loss_mov_rec /= len(val_loader) + 1
                loss_kld /= len(val_loader) + 1
                val_loss /= len(val_loader) + 1
                print('Validation Loss: %.5f Movement Recon Loss: %.5f Motion Recon Loss: %.5f KLD Loss: %.5f:' %
                      (val_loss, loss_mov_rec, loss_mot_rec, loss_kld))

                if epoch % self.opt.eval_every_e == 0:
                    reco_data = self.fake_motions[:4]
                    with torch.no_grad():
                        self.forward(batch_data, 0, schedule_len, eval_mode=True)
                    fake_data = self.fake_motions[:4]
                    gt_data = self.motions[:4]
                    data = torch.cat([fake_data, reco_data, gt_data], dim=0).cpu().numpy()
                    captions = self.caption[:4] * 3
                    save_dir = pjoin(self.opt.eval_dir, 'E%03d_SE%02d_SL%02d'%(epoch, sub_epoch, schedule_len))
                    os.makedirs(save_dir, exist_ok=True)
                    plot_eval(data, save_dir, captions)

                # if cl_ratio == 1:
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    stop_cnt = 0
                elif stop_cnt < self.opt.early_stop_count:
                    stop_cnt += 1
                elif stop_cnt >= self.opt.early_stop_count:
                    break
                if val_loss - min_val_loss >= 0.1:
                    break

            schedule_len += 1

            if schedule_len > 49:
                invalid = False


def main():
    gan_filter = GANTrainer()
    gan_filter.
    print('Hello world')

main()