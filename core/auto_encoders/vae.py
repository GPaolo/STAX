# Created by Giuseppe Paolo 
# Date: 25/02/2021

from core.auto_encoders.base_ae import *
import torch.nn.init as init

class ConvBVAE(BaseAE):
  """
  This class implements a Convolutional B-VAE
  """
  def __init__(self, beta=.001, bias=True, **kwargs):
    self.bias = bias
    self.beta = beta
    self.input_channels = 3
    super(ConvBVAE, self).__init__(**kwargs)
    # Given that we use SELU activations we need the LeCun init, that is the one used by default by Pytorch
    # self.weight_init()

  # ----------------------------------------------------------------
  def weight_init(self):
    for block in self._modules:
      for m in self._modules[block]:
        self.kaiming_init(m)

  def kaiming_init(self, m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      init.kaiming_normal(m.weight, non_linearity='selu')
      if m.bias is not None:
        m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
      m.weight.data.fill_(1)
      if m.bias is not None:
        m.bias.data.fill_(0)
  # ----------------------------------------------------------------

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
                                 nn.Linear(16 * 4 * 4, 2*self.encoding_shape, bias=self.bias), nn.SELU(),
                                 )
    self.feat_mu = nn.Linear(self.encoding_shape, self.encoding_shape, bias=self.bias)
    self.feat_logvar = nn.Linear(self.encoding_shape, self.encoding_shape, bias=self.bias)
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
  def reparametrize(self, mu, logvar):
    """
    Reparametrize the mu and logvar to obtain the features
    :param mu:
    :param logvar:
    :return: mu + std * eps
    """
    std = torch.sqrt(torch.exp(logvar))
    eps = torch.randn_like(std)
    return mu + eps * std
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def encode(self, x):
    h1 = self.encoder(x)
    return self.feat_mu(h1[:, :self.encoding_shape]), self.feat_logvar(h1[:, self.encoding_shape:])
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def forward(self, x):
    """
        Forward pass of the network.
        :param x: Input as RGB array of images
        :return: reconstruction error, features, reconstructed image
        """
    h1 = self.encoder(x)
    mu = self.feat_mu(h1[:, :self.encoding_shape])
    logvar = self.feat_logvar(h1[:, self.encoding_shape:])
    feat = self.reparametrize(mu, logvar)
    y = self.decoder(feat)

    rec_error = self.rec_loss(x, y)
    # Make mean along all the dimensions except the batch one
    dims = list(range(1, len(rec_error.shape)))
    rec_error = torch.mean(rec_error, dim=dims)  # Reconstruction error for each sample
    return {'error':rec_error, 'feat':mu, 'reconstructed':y, 'logvar': logvar} # The features are the mu
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def training_step(self, x):
    """
    Performs one training step of the network
    :param x: Input
    :return: Loss, features, reconstructed image
    """
    self.train()  # Sets network to train mode
    output = self.forward(x)
    # Reconstruction Loss
    rec_loss = torch.mean(output['error'])
    kl_div = self.kl_divergence(output['feat'], output['logvar'])
    loss = rec_loss + self.beta * kl_div

    self.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.eval()  # Sets network to evaluation mode
    return {'total loss': loss.cpu().data, 'rec loss': rec_loss.cpu().data, 'kl div': kl_div.cpu().data}
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def kl_divergence(self, mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
      mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
      logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # total_kld = klds.sum(1).mean(0, True)
    # dimension_wise_kld = klds.mean(0)
    kld = klds.mean(1).mean(0, True)

    return kld[0]#, dimension_wise_kld, mean_kld
  # ----------------------------------------------------------------
