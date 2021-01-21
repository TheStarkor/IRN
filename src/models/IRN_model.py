import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

import models.networks as networks
from .base_model import BaseModel
from models.modules.Quantization import Quantization  # type: ignore
from models.modules.loss import ReconstructionLoss  # type: ignore
import models.lr_scheduler as lr_scheduler

logger = logging.getLogger("base")


class IRNModel(BaseModel):
    def __init__(self, opt: dict):
        super(IRNModel, self).__init__(opt)

        self.train_opt: dict = opt["train"]
        # self.test_opt: str = opt['test']

        # TODO : fix
        self.netG = networks.define_G(opt).to(self.device)
        # self.netG = networks.define_G(opt)
        self.netG = DataParallel(self.netG)

        self.print_network()
        # self.load() TODO

        # TODO
        self.Quantization = Quantization()

        if self.is_train:
            self.netG.train()

            self.Reconstruction_forw = ReconstructionLoss(
                losstype=self.train_opt["pixel_criterion_forw"]
            )
            self.Reconstruction_back = ReconstructionLoss(
                losstype=self.train_opt["pixel_criterion_back"]
            )

            wd_G = (
                self.train_opt["weight_decay_G"]
                if self.train_opt["weight_decay_G"]
                else 0
            )
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning("Params [{:s}] will not optimize.".format(k))
            self.optimizer_G = torch.optim.Adam(
                optim_params,
                lr=self.train_opt["lr_G"],
                weight_decay=wd_G,
                betas=(self.train_opt["beta1"], self.train_opt["beta2"]),
            )
            self.optimizers.append(self.optimizer_G)

            # TODO : scheduler
            if self.train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            self.train_opt["lr_steps"],
                            restarts=self.train_opt["restarts"],
                            weights=self.train_opt["restart_weights"],
                            gamma=self.train_opt["lr_gamma"],
                            clear_state=self.train_opt["clear_state"],
                        )
                    )
            elif self.train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                pass
                # TODO
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.log_dict: dict = OrderedDict()

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str: str = f"{self.netG.__class__.__name__} - {self.netG.module.__class__.__name__}"
        else:
            net_struc_str: str = f"{self.netG.__class__.__name__}"
        logger.info(f"Network G structure: {net_struc_str}, with parameters: {n}")
        logger.info(s)

    def feed_data(self, data):
        # TODO : fix
        self.ref_L = data["LQ"].to(self.device)
        self.real_H = data["GT"].to(self.device)
        # self.ref_L = data["LQ"]
        # self.real_H = data["GT"]

        ### for debugging
        # from torchvision.utils import save_image
        # save_image(self.ref_L, 'img1.png')
        # save_image(self.real_H, 'img2.png')

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y, z):
        l_forw_fit = self.train_opt["lambda_fit_forw"] * self.Reconstruction_forw(
            out, y
        )

        z = z.reshape([out.shape[0], -1])
        l_forw_ce = self.train_opt["lambda_ce_forw"] * torch.sum(z ** 2) / z.shape[0]

        return l_forw_fit, l_forw_ce

    def loss_backward(self, x, y):
        x_samples = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt["lambda_rec_back"] * self.Reconstruction_back(
            x, x_samples_image
        )

        return l_back_rec

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        self.input = self.real_H
        self.output = self.netG(x=self.input)

        zshape = self.output[:, 3:, :, :].shape
        LR_ref = self.ref_L.detach()

        l_forw_fit, l_forw_ce = self.loss_forward(
            self.output[:, :3, :, :], LR_ref, self.output[:, 3:, :, :]
        )

        LR = self.Quantization(self.output[:, :3, :, :])
        # TODO : fix
        gaussian_scale = 1
        y_ = torch.cat((LR, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

        l_back_rec = self.loss_backward(self.real_H, y_)

        loss = l_forw_fit + l_back_rec + l_forw_ce
        loss.backward()

        if self.train_opt["gradient_clipping"]:
            nn.utils.clip_grad_norm_(
                self.netG.parameters(), self.train_opt["gradient_clipping"]
            )

        self.optimizer_G.step()

        self.log_dict["l_forw_fit"] = l_forw_fit.item()
        self.log_dict["l_forw_ce"] = l_forw_ce.item()
        self.log_dict["l_back_rec"] = l_back_rec.item()

        # logger.info(
        #     f"l_forw_fit: {self.log_dict['l_forw_fit']}, l_forw_ce: {self.log_dict['l_forw_ce']}, l_back_rec: {self.log_dict['l_back_rec']}"
        # )

    def test(self):
        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        self.input = self.real_H

        zshape = [
            Lshape[0],
            input_dim * (self.opt["scale"] ** 2) - Lshape[1],
            Lshape[2],
            Lshape[3],
        ]

        gaussian_scale = 1
        # if self.test_opt and self.test_opt['gaussian_scale'] != None:
        #     gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()
        with torch.no_grad():
            self.forw_L = self.netG(x=self.input)[:, :3, :, :]
            self.forw_L = self.Quantization(self.forw_L)
            # TODO: fix
            if (
                self.forw_L.shape[3] != self.gaussian_batch(zshape).shape[3]
                or self.forw_L.shape[2] != self.gaussian_batch(zshape).shape[2]
            ):
                return False
            y_forw = torch.cat(
                (self.forw_L, gaussian_scale * self.gaussian_batch(zshape)), dim=1
            )
            self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["LR_ref"] = self.ref_L.detach()[0].float().cpu()
        out_dict["SR"] = self.fake_H.detach()[0].float().cpu()
        out_dict["LR"] = self.forw_L.detach()[0].float().cpu()
        out_dict["GT"] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def load(self):
        pass
        # load_path_G: str = self.opt['path']['pretrain_model_G']
        # if load_path_G is not None:
        #     logger.info(f'Loading model for G [{load_path_G}] ...')
        # self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
