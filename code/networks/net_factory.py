import torch


def net_factory(
    net_type="unet",
    in_chns=1,
    class_num=3,
    spatial_dims=2,
    device=None,
    **network_kwargs
):
    """Create a 2D legacy model or a 3D volumetric segmentation model.

    Existing 2D behavior is preserved.  Use a ``*_3d`` name, or set
    ``spatial_dims=3`` with a base name, for tensors shaped ``[B,C,D,H,W]``.
    ``device=None`` selects CUDA when available and otherwise uses CPU.
    """
    net_type = net_type.lower()
    if spatial_dims not in (2, 3):
        raise ValueError("spatial_dims must be 2 or 3")
    aliases_3d = {
        "3d_unet": "unet_3d",
        "3d_resunet": "resunet_3d",
        "3d_nnunet": "nnunet_3d",
        "3d_unet_cct": "unet_cct_3d",
        "3d_unet_ctt": "unet_cct_3d",
        "unet_ctt_3d": "unet_cct_3d",
        "unet3d": "unet_3d",
        "resunet3d": "resunet_3d",
        "nnunet3d": "nnunet_3d",
        "unetcct3d": "unet_cct_3d",
        "unetctt3d": "unet_cct_3d",
        "resunet": "resunet_3d",
        "nnunet": "nnunet_3d",
        "unet_ctt": "unet_cct_3d",
    }
    net_type = aliases_3d.get(net_type, net_type)
    if spatial_dims == 3 and not net_type.endswith("_3d"):
        net_type = aliases_3d.get("3d_{}".format(net_type), "{}_3d".format(net_type))

    if net_type == "unet_3d":
        from networks.unet_3d import UNet3D

        net = UNet3D(in_chns=in_chns, class_num=class_num, **network_kwargs)
    elif net_type == "resunet_3d":
        from networks.resunet_3d import ResUNet3D

        net = ResUNet3D(in_chns=in_chns, class_num=class_num, **network_kwargs)
    elif net_type == "nnunet_3d":
        from networks.nnunet_3d import NNUNet3D

        net = NNUNet3D(in_chns=in_chns, class_num=class_num, **network_kwargs)
    elif net_type == "unet_cct_3d":
        from networks.unet_cct_3d import UNetCCT3D

        net = UNetCCT3D(in_chns=in_chns, class_num=class_num, **network_kwargs)
    elif net_type == "unet":
        from networks.unet import UNet

        if network_kwargs:
            raise TypeError("Legacy 2D UNet does not accept extra network arguments")
        net = UNet(in_chns=in_chns, class_num=class_num)
    elif net_type == "unet_hl":
        from networks.unet import UNet_HL

        if network_kwargs:
            raise TypeError("Legacy 2D UNet_HL does not accept extra network arguments")
        net = UNet_HL(in_chns=in_chns, class_num=class_num)
    elif net_type == "unet_lgdt":
        from networks.unet_lgdt import UNet_LGDT

        net = UNet_LGDT(in_chns=in_chns, class_num=class_num, **network_kwargs)
    elif net_type == "unet_cct":
        from networks.unet_cct import UNet_CCT

        if network_kwargs:
            raise TypeError("Legacy 2D UNet_CCT does not accept extra network arguments")
        net = UNet_CCT(in_chns=in_chns, class_num=class_num)
    elif net_type == "mamba_unet":
        from networks.mamba_unet_2d import MambaUNet2D
        net = MambaUNet2D(in_chns=in_chns, class_num=class_num)
    elif net_type == "xnetv2":
        from networks.xnetv2 import XNetv2
        net = XNetv2(in_channels=in_chns, num_classes=class_num)
    elif net_type == "quanmambascrib":
        from networks.quan_mamba_scrib import QuanMambaScrib
        net = QuanMambaScrib(
            in_chns=in_chns,
            class_num=class_num,
            unet_type="unet_hl",
            mamba_variant="vmunet",
            qpim_backend="torch_angle_fidelity",
            ignore_index=class_num,
        )
    else:
        raise ValueError(f"Unknown net_type: {net_type}")
    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    net = net.to(target_device)
    return net
