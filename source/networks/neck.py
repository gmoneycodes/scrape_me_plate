class YOLOv7NECK(nn.Module):
    """
    Only proceed 3 layer input. Like stage2, stage3, stage4.
    """

    def __init__(
            self,
            depths=(1, 1, 1, 1),
            in_channels=(512, 1024, 1024),
            norm='bn',
            act="silu",
    ):
        super().__init__()

        # top-down conv
        self.spp = SPPCSPC(in_channels[2], in_channels[2] // 2, k=(5, 9, 13))
        self.conv_for_P5 = BaseConv(in_channels[2] // 2, in_channels[2] // 4, 1, 1, norm=norm, act=act)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_for_C4 = BaseConv(in_channels[1], in_channels[2] // 4, 1, 1, norm=norm, act=act)
        self.p5_p4 = CSPLayer(
            in_channels[2] // 2,
            in_channels[2] // 4,
            expansion=0.5,
            num_bottle=depths[0],
            norm=norm,
            act=act,
            )

        self.conv_for_P4 = BaseConv(in_channels[2] // 4, in_channels[2] // 8, 1, 1, norm=norm, act=act)
        self.conv_for_C3 = BaseConv(in_channels[0], in_channels[2] // 8, 1, 1, norm=norm, act=act)
        self.p4_p3 = CSPLayer(
            in_channels[2] // 4,
            in_channels[2] // 8,
            expansion=0.5,
            num_bottle=depths[0],
            norm=norm,
            act=act,
            )

        # bottom-up conv
        self.downsample_conv1 = Transition(in_channels[2] // 8, in_channels[2] // 4, mpk=2, norm=norm, act=act)
        self.n3_n4 = CSPLayer(
            in_channels[2] // 2,
            in_channels[2] // 4,
            expansion=0.5,
            num_bottle=depths[0],
            norm=norm,
            act=act,
            )

        self.downsample_conv2 = Transition(in_channels[2] // 4, in_channels[2] // 2, mpk=2, norm=norm, act=act)
        self.n4_n5 = CSPLayer(
            in_channels[2],
            in_channels[2] // 2,
            expansion=0.5,
            num_bottle=depths[0],
            norm=norm,
            act=act,
            )

        self.n3 = BaseConv(in_channels[2] // 8, in_channels[2] // 4, 3, 1, norm=norm, act=act)
        self.n4 = BaseConv(in_channels[2] // 4, in_channels[2] // 2, 3, 1, norm=norm, act=act)
        self.n5 = BaseConv(in_channels[2] // 2, in_channels[2], 3, 1, norm=norm, act=act)

    def forward(self, inputs):
        #  backbone
        [c3, c4, c5] = inputs
        # top-down
        p5 = self.spp(c5)
        p5_shrink = self.conv_for_P5(p5)
        p5_upsample = self.upsample(p5_shrink)
        p4 = torch.cat([p5_upsample, self.conv_for_C4(c4)], 1)
        p4 = self.p5_p4(p4)

        p4_shrink = self.conv_for_P4(p4)
        p4_upsample = self.upsample(p4_shrink)
        p3 = torch.cat([p4_upsample, self.conv_for_C3(c3)], 1)
        p3 = self.p4_p3(p3)

        # down-top
        n3 = p3
        n3_downsample = self.downsample_conv1(n3)
        n4 = torch.cat([n3_downsample, p4], 1)
        n4 = self.n3_n4(n4)

        n4_downsample = self.downsample_conv2(n4)
        n5 = torch.cat([n4_downsample, p5], 1)
        n5 = self.n4_n5(n5)

        n3 = self.n3(n3)
        n4 = self.n4(n4)
        n5 = self.n5(n5)

        outputs = (n3, n4, n5)
        return outputs