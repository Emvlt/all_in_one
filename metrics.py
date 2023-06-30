from torch.nn import MSELoss
from torch import Tensor, log10

class PSNR():
    def __init__(self):
        self.loss = MSELoss()

    def __call__(self, inferred:Tensor, target:Tensor) -> Tensor:
        return 20 * log10(target.max()-target.min()) - 10 * log10(self.loss(inferred, target))
