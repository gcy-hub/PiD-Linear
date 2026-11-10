"""Check calibrated attention outputs, gradients and unchanged text-query rows."""
import torch
from .mixing import biased_attention
from flux3_qdm.sliding import sliding_joint_attention


def main():
    torch.manual_seed(7)
    grid=(8,8);text=9
    cpu=[torch.randn(1,2,73,32,requires_grad=True) for _ in range(3)]
    bias=torch.tensor([-1.,-2.])
    original=sliding_joint_attention(*cpu,grid,text,3)
    zero=biased_attention(*cpu,grid,text,torch.zeros(2),3)
    assert torch.allclose(original,zero,atol=1e-6)
    corrected=biased_attention(*cpu,grid,text,bias,3)
    assert torch.allclose(original[:,:,64:],corrected[:,:,64:],atol=1e-6)
    corrected.square().sum().backward()
    gpu=[x.detach().cuda().requires_grad_() for x in cpu]
    actual=biased_attention(*gpu,grid,text,bias.cuda(),3)
    actual.square().sum().backward()
    torch.testing.assert_close(actual.cpu(),corrected,rtol=2e-4,atol=2e-5)
    for a,b in zip(cpu,gpu):torch.testing.assert_close(b.grad.cpu(),a.grad,rtol=3e-3,atol=2e-4)
    print('Zero-bias identity, unchanged text queries, CUDA outputs and gradients passed')


if __name__=='__main__':main()
