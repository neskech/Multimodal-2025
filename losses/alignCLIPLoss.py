from AlignCLIP.align_clip.loss import ClipInModalityLoss
class AlignCLIPLoss:
    def __init__(self):
        self.criterion = ClipInModalityLoss(alpha = 1.0, beta = 0.5, nl_semantic_supervision=True)

    def forward(self, image_features, text_features, logits_scale, semantic_features=None):
        # Compute the alignment loss between image and text features
        return self.criterion(image_features, text_features, logits_scale, semantic_features=semantic_features)