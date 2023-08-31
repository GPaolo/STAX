# Created by Giuseppe Paolo 
# Date: 16/02/2021

from core.auto_encoders.base_ae import *

# ----------------------------------------------------------------
class FFAE(BaseAE):
  """
  This class implement the FeedForward Autoencoder
  """

  # ----------------------------------------------------------------
  def _define_encoder(self):
    """
    Defines encoder of the network
    """
    self.encoder = nn.Sequential(View((-1, 64 * 64 * 3)),
                                 nn.Linear(64 * 64 * 3, 2560, bias=False), nn.SELU(),
                                 nn.Linear(2560, 128, bias=False), nn.SELU(),
                                 nn.Linear(128, self.encoding_shape, bias=False), nn.SELU(),
                                 )

  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def _define_decoder(self):
    """
    Defines decoder of the network
    """
    self.decoder = nn.Sequential(nn.Linear(self.encoding_shape, 128, bias=False), nn.SELU(),
                                 nn.Linear(128, 2560, bias=False), nn.SELU(),
                                 nn.Linear(2560, 64 * 64 * 3, bias=False), nn.ReLU(),
                                 View((-1, 3, 64, 64)),
                                 )
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def forward(self, x):
    """
        Forward pass of the network.
        :param x: Input as RGB array of images
        :return: reconstruction error, features, reconstructed image
        """
    feat = self.encoder(x)
    y = self.decoder(feat)

    rec_error = self.rec_loss(x, y)
    # Make mean along all the dimensions except the batch one
    dims = list(range(1, len(rec_error.shape)))
    rec_error = torch.mean(rec_error, dim=dims)  # Reconstruction error for each sample
    return rec_error, torch.squeeze(feat), y
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def training_step(self, x):
    """
    Performs one training step of the network
    :param x: Input
    :return: Loss, features, reconstructed image
    """
    self.train() # Sets network to train mode
    rec_error, feat, y = self.forward(x)
    # Reconstruction Loss
    loss = torch.mean(rec_error)

    self.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.eval() # Sets network to evaluation mode
    return loss.cpu().data
  # ----------------------------------------------------------------
# ----------------------------------------------------------------


# ----------------------------------------------------------------
class ConvAE(BaseAE):
  """
  This class implements the Convolutional Autoencoder
  """
  def __init__(self, bias=True, **kwargs):
    self.bias = bias
    self.input_channels = 3
    super(ConvAE, self).__init__(**kwargs)

  # ----------------------------------------------------------------
  def _define_encoder(self):
    """
    Define encoder
    :return:
    """
    # No need to use BatchNorm with SELU cause SELU already normalizes the net
    self.encoder = nn.Sequential(nn.Conv2d(self.input_channels, 32, kernel_size=4, stride=2, padding=1, bias=self.bias), nn.SELU(),  # 64->32
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=self.bias), nn.SELU(), # 32->16
                                 nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1, bias=self.bias), nn.SELU(), # 16->8
                                 nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1, bias=self.bias), nn.SELU(),  # 8->4
                                 View((-1, 16 * 4 * 4)),
                                 nn.Linear(16 * 4 * 4, self.encoding_shape, bias=self.bias), nn.SELU(),
                                 )
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def _define_decoder(self):
    """
    Defines the decoder
    """
    self.decoder = nn.Sequential(nn.Linear(self.encoding_shape, 16 * 4 * 4, bias=self.bias), nn.SELU(),
                                 View((-1, 16, 4, 4)),
                                 nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1, bias=self.bias), nn.SELU(), # 4 -> 8
                                 nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1, bias=self.bias), nn.SELU(), # 8 -> 16
                                 nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=self.bias), nn.SELU(), # 16 -> 32
                                 nn.ConvTranspose2d(32, self.input_channels, kernel_size=4, stride=2, padding=1, bias=self.bias), nn.ReLU(),  # 32 -> 64
                                 )
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def forward(self, x):
    """
        Forward pass of the network.
        :param x: Input as RGB array of images
        :return: reconstruction error, features, reconstructed image
        """
    feat = self.encoder(x)
    y = self.decoder(feat)

    rec_error = self.rec_loss(x, y)
    # Make mean along all the dimensions except the batch one
    dims = list(range(1, len(rec_error.shape)))
    rec_error = torch.mean(rec_error, dim=dims)  # Reconstruction error for each sample
    return {'error':rec_error, 'feat':torch.squeeze(feat), 'reconstructed':y}
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def training_step(self, x):
    """
    Performs one training step of the network
    :param x: Input
    :return: Loss, features, reconstructed image
    """
    self.train() # Sets network to train mode
    output = self.forward(x)
    # Reconstruction Loss
    loss = torch.mean(output['error'])

    self.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.eval() # Sets network to evaluation mode
    return {'total loss': loss.cpu().data, 'rec loss': loss.cpu().data, 'kl div': None}
  # ----------------------------------------------------------------
# ----------------------------------------------------------------