import torch

import matplotlib.pyplot as plt
from tqdm import tqdm

from models.ittt.modelling_ittt import ItttModel
from utils import constants

CHECKPOINT = "aklein4/iTTT_longattn-precond-init-norm"
STEP = 555


@torch.no_grad()
def main():
    
    model = ItttModel.from_pretrained(
        CHECKPOINT,
        subfolder=f"step_{STEP:012d}",
        _attn_implementation="flash_attention_2"
    )

    p_r = model.llama.model.layers[16].mlp.fast.p_l.data.to(constants.DEVICE)

    s, v = torch.linalg.eig(p_r.float())

    print(s)
    return
    
    lr = model.llama.model.layers[16].mlp.fast.get_lr().to(constants.DEVICE)

    # normalize rows and columns using sinkhorn iterations
    for _ in tqdm(range(20)):
        lr = lr / lr.sum(dim=0, keepdim=True)
        lr = lr / lr.sum(dim=1, keepdim=True)

    lr = lr / torch.norm(lr, dim=1, keepdim=True)

    # sort rows by similarity
    # curr = lr[0].clone()
    # lr[0, 0] += 1000
    # rows = [curr]
    # ids = [0]
    # for i in tqdm(range(1, lr.shape[0])):

    #     sims = (lr / torch.norm(lr, dim=1, keepdim=True)) @ curr
    #     idx = torch.argsort(sims, descending=True)
    #     id = idx[0].item()

    #     curr = lr[id].clone()
    #     rows.append(curr)

    #     lr[id, 0] += 1000
        # ids.append(id)

    # ids.sort()
    # print(ids)

    # lr = torch.stack(rows)

    sims = lr @ lr[0]
    idx = torch.argsort(sims, descending=True)

    lr = lr[idx]

    sims = lr @ lr.T

    plt.matshow(torch.log10(lr).cpu())
    # plt.matshow(sims.cpu())
    plt.colorbar()

    plt.savefig("lr_probe.png")


if __name__ == "__main__":
    main()
