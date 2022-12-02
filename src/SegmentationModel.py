import segmentation_models_pytorch as smp

class SegmentationModel:
    def __init__(self):
        self.encoder_name = "resnet50"
        self.encoder_depth = 5
        self.encoder_weights = "imagenet"
        self.decoder_channels = (256,128,64,16,4)
        self.decoder_use_batchnorm = False
        self.decoder_attention_type = None
        self.in_channels = 3 #(Only RGB Channels)
        self.classes = 5
        self.activation  = "softmax"
        self.aux_params = None
        self.model = None

    def InitializeModel(self):

        self.model = smp.Unet(
                               encoder_name= self.encoder_name,
                                encoder_depth = self.encoder_depth,
                                encoder_weights = self.encoder_weights,
                                decoder_channels = self.decoder_channels,
                                decoder_use_batchnorm = self.decoder_use_batchnorm,
                                decoder_attention_type = self.decoder_attention_type,
                                in_channels = self.in_channels,
                                classes = self.classes,
                                activation = self.activation,
                                aux_params= self.aux_params)
