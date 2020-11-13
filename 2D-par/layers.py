import torch

from distdl.utilities.torch import NoneTensor


class DistributedNetworkOutputFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input, partition):
		
		ctx.partition = partition

		if partition.rank == 0:
			return input.clone()

		else:
			return torch.tensor([0.0], requires_grad=True).float()

	@staticmethod
	def backward(ctx, grad_output):

		partition = ctx.partition

		if partition.rank == 0:

		else:
			return NoneTensor(), None

class DistributedNetworkOutput(torch.nn.Module):

	def __init__(self, partition):
		super(DistributedNetworkOutput, self).__init__()
		self.partition = partition

	def forward(self, input):
		return DistributedNetworkOutputFunction.apply(input, self.partition)

# Set of layers implemented in Unet that occurs frequently
def dist_dual_conv(P_conv, in_channel, out_channel):
	conv = torch.nn.Sequential(
		distdl.nn.DistributedConv2d(P_conv,
					    in_channels = in_channel,
					    out_channels = out_channel,
					    kernel_size=(3,3),
					    padding=(1,1)),
		# REPLACE W UPSCALING LATER
		distdl.nn.DistributedBatchNorm(P_conv, num_features=1),
		torch.nn.ReLU(),
		distdl.nn.DistributedConv2d(P_conv,
                                            in_channels = in_channel,
                                            out_channels = out_channel,
                                            kernel_size=(3,3),
                                            padding=(1,1)),
		# REPLACE W UPSCALING LATER
		distdl.nn.DistributedBatchNorm(P_conv, num_features=1),
		torch.nn.ReLU()
	)

	return conv
