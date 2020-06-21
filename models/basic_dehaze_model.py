import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F

from util import util


class BasicDehazeModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_haze', type=float, default=0.1, help='weight for D_haze')
            parser.add_argument('--lambda_clear', type=float, default=0.1, help='weight for D_clear')
            parser.add_argument('--lambda_tv', type=float, default=1, help='weight for D_clear')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        
        parser.add_argument('--netR_T', type=str, default='unet_trans_256', help='specify generator architecture')
        parser.add_argument('--netR_J', type=str, default='haze_refine_2', help='specify generator architecture')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_haze', 'G_rec_I', 'D_clear', 'G_ref_J', 'rec_I', 'rec_J', 'TV_T', 'idt_J']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_I', 'est_J', 'rec_I', 'rec_J',
                            'est_T_vis', 'out_T_vis', 'real_J']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['Est_T', 'Est_J', 'D_haze', 'D_clear']
        else:  # during test time, only load Gs
            self.model_names = ['Est_T', 'Est_J']

        # define networks (both Generators and discriminators)
        self.netEst_T = networks.define_G(opt.input_nc, 1, opt.ngf, opt.netR_T, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netEst_J = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netR_J, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_haze = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_clear = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
 
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_I_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_J_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionRec = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionTV = networks.TVLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netEst_T.parameters(), self.netEst_J.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_haze.parameters(), self.netD_clear.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_I = input['haze'].to(self.device) # [-1, 1]
        self.real_J = input['clear'].to(self.device) # [-1, 1]
        self.image_paths = input['paths']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # output scale [0,1]
        self.est_T, self.out_T = self.netEst_T(self.real_I)
        self.est_J = self.netEst_J(self.real_I)

        # reconstruct haze image
        est_T_map = self.est_T.repeat(1,3,1,1)
        self.rec_I = util.synthesize_fog(self.est_J, est_T_map)
        self.rec_J = util.reverse_fog(self.real_I, est_T_map)


    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """

        with torch.no_grad():
            self.forward()
            self.compute_visuals()
            self.refine_J = (self.rec_J + self.est_J)/2
            self.visual_names.append('refine_J')

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        # rescale to [-1,1] for visdom 
        self.est_T_vis = (self.est_T - 0.5)/0.5
        self.out_T_vis = (self.out_T - 0.5)/0.5
        # self.map_A_vis = (self.map_A - 0.5)/0.5


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_haze(self):
        fake_I = self.fake_I_pool.query(self.rec_I)
        self.loss_D_haze = self.backward_D_basic(self.netD_haze, self.real_I, fake_I)

    def backward_D_clear(self):
        fake_J = self.fake_J_pool.query(self.est_J)
        self.loss_D_clear = self.backward_D_basic(self.netD_clear, self.real_J, fake_J)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_tv = self.opt.lambda_tv
        lambda_haze = self.opt.lambda_haze
        lambda_clear = self.opt.lambda_clear

        # TV loss
        if lambda_tv > 0.0:
            self.loss_TV_T = self.criterionTV(self.out_T)*lambda_tv
        else:
            self.loss_TV_T = 0

        # Identity loss
        if lambda_idt > 0.0:
            self.loss_idt_J = self.criterionIdt(self.netEst_J(self.real_J), self.real_J)*lambda_idt
        else:
            self.loss_idt_J = 0

        # Generator losses for rec_I and est_J
        self.loss_G_rec_I = self.criterionGAN(self.netD_haze(self.rec_I), True)*lambda_haze
        self.loss_G_ref_J = self.criterionGAN(self.netD_clear(self.est_J), True)*lambda_clear #+ \
#                            self.criterionGAN(self.netD_clear(self.rec_J), True)*lambda_clear

        # Reconstrcut loss
        self.loss_rec_I = self.criterionRec(self.rec_I, self.real_I)
        # only compute, not back propagate
        self.loss_rec_J = self.criterionRec(self.rec_J, self.est_J)
        
        self.loss_G = self.loss_G_rec_I + self.loss_G_ref_J + self.loss_rec_I + self.loss_idt_J + self.loss_TV_T
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_haze, self.netD_clear], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_haze, self.netD_clear], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_haze()      # calculate gradients for D_A
        self.backward_D_clear()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
