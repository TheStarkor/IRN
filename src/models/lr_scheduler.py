import math
import torch
from collections import Counter, defaultdict
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional


class MultiStepLR_Restart(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones: List[int],
        restarts: Optional[List[int]] = None,
        weights: Optional[List[int]] = None,
        gamma: float = 0.1,
        clear_state: bool = False,
        last_epoch: int = -1,
    ):
        self.milestones: dict = Counter(milestones)
        self.gamma: float = gamma
        self.clear_state: bool = clear_state
        self.restarts: List[int] = restarts if restarts else [0]
        self.restarts_weights: List[int] = weights if weights else [1]

        assert len(self.restarts) == len(
            self.restarts_weights
        ), "restarts and their weights do not match."

        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight: int = self.restarts_weights[self.restarts.index(self.last_epoch)]
            return [
                group["initial_lr"] * weight for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


if __name__ == "__main__":
    optimizer = torch.optim.Adam(
        [torch.zeros(3, 64, 3, 3)], lr=2e-4, weight_decay=0, betas=(0.9, 0.99)
    )
    ##############################
    # MultiStepLR_Restart
    ##############################
    ## Original
    lr_steps = [200000, 400000, 600000, 800000]
    restarts = None
    restart_weights = None

    ## two
    lr_steps = [
        100000,
        200000,
        300000,
        400000,
        490000,
        600000,
        700000,
        800000,
        900000,
        990000,
    ]
    restarts = [500000]
    restart_weights = [1]

    ## four
    lr_steps = [
        50000,
        100000,
        150000,
        200000,
        240000,
        300000,
        350000,
        400000,
        450000,
        490000,
        550000,
        600000,
        650000,
        700000,
        740000,
        800000,
        850000,
        900000,
        950000,
        990000,
    ]
    restarts = [250000, 500000, 750000]
    restart_weights = [1, 1, 1]

    scheduler = MultiStepLR_Restart(
        optimizer, lr_steps, restarts, restart_weights, gamma=0.5, clear_state=False
    )

    N_iter = 1000000
    lr_l = list(range(N_iter))
    for i in range(N_iter):
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        lr_l[i] = current_lr

    import matplotlib as mpl  # type: ignore
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick  # type: ignore

    mpl.style.use("default")
    import seaborn  # type: ignore

    seaborn.set(style="whitegrid")
    seaborn.set_context("paper")

    plt.figure(1)
    plt.subplot(111)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    plt.title("Title", fontsize=16, color="k")
    plt.plot(list(range(N_iter)), lr_l, linewidth=1.5, label="learning rate scheme")
    legend = plt.legend(loc="upper right", shadow=False)
    ax = plt.gca()
    labels = ax.get_xticks().tolist()
    for k, v in enumerate(labels):
        labels[k] = str(int(v / 1000)) + "K"
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))

    ax.set_ylabel("Learning rate")
    ax.set_xlabel("Iteration")
    fig = plt.gcf()
    plt.savefig("fig1.png", dpi=300)
