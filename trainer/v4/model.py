"""DynUNet backbone shared by every v4 arm: the nnU-Net 3d_fullres-like configuration of
trainer/npc_sota_train.build_model('nnunet'), with variable input / output channels."""
from monai.networks.nets import DynUNet


def build_dynunet(in_channels=1, out_channels=2):
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2]] * 5,
        filters=[32, 64, 128, 256, 320, 320],
        res_block=True,
    )


def stage2_channels(cfg):
    """(in_channels, out_channels) of a Stage-2 config: CT + prior channels; Y (2) + visibility head (5)."""
    n_in = 1 + len(cfg.get("priors", []))
    n_out = 2 + (5 if cfg.get("vis_head", {}).get("enabled") else 0)
    return n_in, n_out
