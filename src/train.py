import argparse
import logging
import torch
import math
from typing import Dict, Union, Any
import os

import options.options as option
from utils import util
from data import create_dataset, create_dataloader
from models import create_model

PATH = "training"
NAME = "testing"
MANUAL_SEED = 10


def main():
    ### parser
    ### diff : cannot support distribution
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to YAML file.")
    args = parser.parse_args()
    opt: Dict[str, Any] = option.parse(args.opt, is_train=True)

    ### mkdir and loggers
    util.mkdir_and_rename(
        opt["path"]["experiments_root"]
    )  # rename experiment folder if exists
    util.mkdirs(
        (
            path
            for key, path in opt["path"].items()
            if not key == "experiments_root"
            and "pretrain_model" not in key
            and "resume" not in key
        )
    )

    util.setup_logger(
        "base", PATH, "train_" + NAME, level=logging.INFO, screen=True, tofile=False
    )
    util.setup_logger(
        "val", PATH, "val_" + NAME, level=logging.INFO, screen=True, tofile=False
    )

    logger: Logger = logging.getLogger("base")

    opt = option.dict_to_nonedict(opt)

    # TODO : tensorboard logger

    ### random seed
    seed: int = MANUAL_SEED
    logger.info(f"Random seed: {seed}")
    util.set_random_seed(seed)

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2
    torch.backends.cudnn.benchmark = True

    ### create train and val dataloader
    phase: str
    dataset_opt: Dict[str, Any]
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set: Union[LQGTDataset] = create_dataset(dataset_opt)
            train_size: int = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters: int = int(opt["train"]["niter"])
            total_epochs: int = int(math.ceil(total_iters / train_size))

            train_loader = create_dataloader(train_set, dataset_opt, opt, None)

            logger.info(
                "Number of train images: {:,d}, iters: {:,d}".format(
                    len(train_set), train_size
                )
            )
            logger.info(
                "Total epochs needed: {:d} for iters {:,d}".format(
                    total_epochs, total_iters
                )
            )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
        else:
            raise NotImplementedError(f"Phase [{phase:s}] is not recognized")

    model = create_model(opt)

    current_step: int = 0
    start_epoch: int = 0

    # TODO : training
    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_step}")
    # for epoch in range(start_epoch, total_epochs + 1):
    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            ### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = f"<epoch:{epoch:3d}, iter:{current_step:8d}, lr:{model.get_current_learning_rate():3e}> "
                for k, v in logs.items():
                    message += f"{k:s}: {v:.4e} "

                    # TODO: tensorboard
                logger.info(message)

            # validation
            if current_step % opt["train"]["val_freq"] == 0:
                avg_psnr: float = 0.0
                idx: int = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(
                        os.path.basename(val_data["LQ_path"][0])
                    )[0]
                    img_dir = os.path.join(opt["path"]["val_images"], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    # TODO: fix
                    if model.test() == False:
                        continue

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals["SR"])  # uint8
                    gt_img = util.tensor2img(visuals["GT"])  # uint8

                    lr_img = util.tensor2img(visuals["LR"])

                    gtl_img = util.tensor2img(visuals["LR_ref"])

                    # Save SR images for reference
                    save_img_path = os.path.join(
                        img_dir, "{:s}_{:d}.png".format(img_name, current_step)
                    )
                    util.save_img(sr_img, save_img_path)

                    # Save LR images
                    save_img_path_L = os.path.join(
                        img_dir, "{:s}_forwLR_{:d}.png".format(img_name, current_step)
                    )
                    util.save_img(lr_img, save_img_path_L)

                    # Save ground truth
                    if current_step == opt["train"]["val_freq"]:
                        save_img_path_gt = os.path.join(
                            img_dir, "{:s}_GT_{:d}.png".format(img_name, current_step)
                        )
                        util.save_img(gt_img, save_img_path_gt)
                        save_img_path_gtl = os.path.join(
                            img_dir,
                            "{:s}_LR_ref_{:d}.png".format(img_name, current_step),
                        )
                        util.save_img(gtl_img, save_img_path_gtl)

                    # calculate PSNR
                    crop_size = opt["scale"]
                    gt_img = gt_img / 255.0
                    sr_img = sr_img / 255.0
                    cropped_sr_img = sr_img[
                        crop_size:-crop_size, crop_size:-crop_size, :
                    ]
                    cropped_gt_img = gt_img[
                        crop_size:-crop_size, crop_size:-crop_size, :
                    ]
                    avg_psnr += util.calculate_psnr(
                        cropped_sr_img * 255, cropped_gt_img * 255
                    )

                avg_psnr = avg_psnr / idx

                # log
                logger.info("# Validation # PSNR: {:.4e}.".format(avg_psnr))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}.".format(
                        epoch, current_step, avg_psnr
                    )
                )
                # TODO: tensorboard

            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
                model.save(current_step)
                model.save_training_state(epoch, current_step)


if __name__ == "__main__":
    main()
