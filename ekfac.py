import torch
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer


class EKFAC(Optimizer):

    def __init__(self, net, eps, sua=False, ra=False, update_freq=1,
                 alpha=.75):
        """ EKFAC Preconditionner for Linear and Conv2d layers.

        Computes the EKFAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            ra (bool): Computes stats using a running average of averaged gradients
                instead of using a intra minibatch estimate
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter

        """
        self.eps = eps
        self.sua = sua
        self.ra = ra
        self.update_freq = update_freq
        self.alpha = alpha
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        if not self.ra and self.alpha != 1.:
            raise NotImplementedError
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                if mod_class == 'Conv2d':
                    if not self.sua:
                        # Adding gathering filter for convolution
                        d['gathering_filter'] = self._get_gathering_filter(mod)
                self.params.append(d)
        super(EKFAC, self).__init__(self.params, {})

    def step(self, update_stats=True, update_params=True):
        """Performs one step of preconditioning."""
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            # Update convariances and inverses
            if self._iteration_counter % self.update_freq == 0:
                self._compute_kfe(group, state)
            # Preconditionning
            if group['layer_type'] == 'Conv2d' and self.sua:
                if self.ra:
                    self._precond_sua_ra(weight, bias, group, state)
                else:
                    self._precond_intra_sua(weight, bias, group, state)
            else:
                if self.ra:
                    self._precond_ra(weight, bias, group, state)
                else:
                    self._precond_intra(weight, bias, group, state)
        self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _precond_ra(self, weight, bias, group, state):
        """Applies preconditioning."""
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        m2 = state['m2']
        g = weight.grad.data
        s = g.shape
        bs = self.state[group['mod']]['x'].size(0)
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        g_kfe = torch.mm(torch.mm(kfe_gy.t(), g), kfe_x)
        m2.mul_(self.alpha).add_((1. - self.alpha) * bs, g_kfe**2)
        g_nat_kfe = g_kfe / (m2 + self.eps)
        g_nat = torch.mm(torch.mm(kfe_gy, g_nat_kfe), kfe_x.t())
        if bias is not None:
            gb = g_nat[:, -1].contiguous().view(*bias.shape)
            bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        g_nat = g_nat.contiguous().view(*s)
        weight.grad.data = g_nat

    def _precond_intra(self, weight, bias, group, state):
        """Applies preconditioning."""
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        mod = group['mod']
        x = self.state[mod]['x']
        gy = self.state[mod]['gy']
        g = weight.grad.data
        s = g.shape
        s_x = x.size()
        s_cin = 0
        s_gy = gy.size()
        bs = x.size(0)
        if group['layer_type'] == 'Conv2d':
            x = F.conv2d(x, group['gathering_filter'],
                         stride=mod.stride, padding=mod.padding,
                         groups=mod.in_channels)
            s_x = x.size()
            x = x.data.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            if mod.bias is not None:
                ones = torch.ones_like(x[:1])
                x = torch.cat([x, ones], dim=0)
                s_cin = 1 # adding a channel in dim for the bias
            # intra minibatch m2
            x_kfe = torch.mm(kfe_x.t(), x).view(s_x[1]+s_cin, -1, s_x[2], s_x[3]).permute(1, 0, 2, 3)
            gy = gy.permute(1, 0, 2, 3).contiguous().view(s_gy[1], -1)
            gy_kfe = torch.mm(kfe_gy.t(), gy).view(s_gy[1], -1, s_gy[2], s_gy[3]).permute(1, 0, 2, 3)
            m2 = torch.zeros((s[0], s[1]*s[2]*s[3]+s_cin), device=g.device)
            g_kfe = torch.zeros((s[0], s[1]*s[2]*s[3]+s_cin), device=g.device)
            for i in range(x_kfe.size(0)):
                g_this = torch.mm(gy_kfe[i].view(s_gy[1], -1),
                                  x_kfe[i].permute(1, 2, 0).view(-1, s_x[1]+s_cin))
                m2 += g_this**2
            m2 /= bs
            g_kfe = torch.mm(gy_kfe.permute(1, 0, 2, 3).view(s_gy[1], -1),
                             x_kfe.permute(0, 2, 3, 1).contiguous().view(-1, s_x[1]+s_cin)) / bs
            ## sanity check did we obtain the same grad ?
            # g = torch.mm(torch.mm(kfe_gy, g_kfe), kfe_x.t())
            # gb = g[:,-1]
            # gw = g[:,:-1].view(*s)
            # print('bias', torch.dist(gb, bias.grad.data))
            # print('weight', torch.dist(gw, weight.grad.data))
            ## end sanity check
            g_nat_kfe = g_kfe / (m2 + self.eps)
            g_nat = torch.mm(torch.mm(kfe_gy, g_nat_kfe), kfe_x.t())
            if bias is not None:
                gb = g_nat[:, -1].contiguous().view(*bias.shape)
                bias.grad.data = gb
                g_nat = g_nat[:, :-1]
            g_nat = g_nat.contiguous().view(*s)
            weight.grad.data = g_nat
        else:
            if bias is not None:
                ones = torch.ones_like(x[:, :1])
                x = torch.cat([x, ones], dim=1)
            x_kfe = torch.mm(x, kfe_x)
            gy_kfe = torch.mm(gy, kfe_gy)
            m2 = torch.mm(gy_kfe.t()**2, x_kfe**2) / bs
            g_kfe = torch.mm(gy_kfe.t(), x_kfe) / bs
            g_nat_kfe = g_kfe / (m2 + self.eps)
            g_nat = torch.mm(torch.mm(kfe_gy, g_nat_kfe), kfe_x.t())
            if bias is not None:
                gb = g_nat[:, -1].contiguous().view(*bias.shape)
                bias.grad.data = gb
                g_nat = g_nat[:, :-1]
            g_nat = g_nat.contiguous().view(*s)
            weight.grad.data = g_nat

    def _precond_sua_ra(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        m2 = state['m2']
        g = weight.grad.data
        s = g.shape
        bs = self.state[group['mod']]['x'].size(0)
        mod = group['mod']
        if bias is not None:
            gb = bias.grad.view(-1, 1, 1, 1).expand(-1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=1)
        g_kfe = self._to_kfe_sua(g, kfe_x, kfe_gy)
        m2.mul_(self.alpha).add_((1. - self.alpha) * bs, g_kfe**2)
        g_nat_kfe = g_kfe / (m2 + self.eps)
        g_nat = self._to_kfe_sua(g_nat_kfe, kfe_x.t(), kfe_gy.t())
        if bias is not None:
            gb = g_nat[:, -1, s[2]//2, s[3]//2]
            bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        weight.grad.data = g_nat

    def _precond_intra_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        mod = group['mod']
        x = self.state[mod]['x']
        gy = self.state[mod]['gy']
        g = weight.grad.data
        s = g.shape
        s_x = x.size()
        s_gy = gy.size()
        s_cin = 0
        bs = x.size(0)
        if bias is not None:
            ones = torch.ones_like(x[:,:1])
            x = torch.cat([x, ones], dim=1)
            s_cin += 1
        # intra minibatch m2
        x = x.permute(1, 0, 2, 3).contiguous().view(s_x[1]+s_cin, -1)
        x_kfe = torch.mm(kfe_x.t(), x).view(s_x[1]+s_cin, -1, s_x[2], s_x[3]).permute(1, 0, 2, 3)
        gy = gy.permute(1, 0, 2, 3).contiguous().view(s_gy[1], -1)
        gy_kfe = torch.mm(kfe_gy.t(), gy).view(s_gy[1], -1, s_gy[2], s_gy[3]).permute(1, 0, 2, 3)
        m2 = torch.zeros((s[0], s[1]+s_cin, s[2], s[3]), device=g.device)
        g_kfe = torch.zeros((s[0], s[1]+s_cin, s[2], s[3]), device=g.device)
        for i in range(x_kfe.size(0)):
            g_this = grad_wrt_kernel(x_kfe[i:i+1], gy_kfe[i:i+1], mod.padding, mod.stride)
            m2 += g_this**2
        m2 /= bs
        g_kfe = grad_wrt_kernel(x_kfe, gy_kfe, mod.padding, mod.stride) / bs
        ## sanity check did we obtain the same grad ?
        # g = self._to_kfe_sua(g_kfe, kfe_x.t(), kfe_gy.t())
        # gb = g[:, -1, s[2]//2, s[3]//2]
        # gw = g[:,:-1].view(*s)
        # print('bias', torch.dist(gb, bias.grad.data))
        # print('weight', torch.dist(gw, weight.grad.data))
        ## end sanity check
        g_nat_kfe = g_kfe / (m2 + self.eps)
        g_nat = self._to_kfe_sua(g_nat_kfe, kfe_x.t(), kfe_gy.t())
        if bias is not None:
            gb = g_nat[:, -1, s[2]//2, s[3]//2]
            bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        weight.grad.data = g_nat

    def _compute_kfe(self, group, state):
        """Computes the covariances."""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                x = F.conv2d(x, group['gathering_filter'],
                             stride=mod.stride, padding=mod.padding,
                             groups=mod.in_channels)
            x = x.data.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        xxt = torch.mm(x, x.t()) / float(x.shape[1])
        Ex, state['kfe_x'] = torch.symeig(xxt, eigenvectors=True)
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1
        ggt = torch.mm(gy, gy.t()) / float(gy.shape[1])
        Eg, state['kfe_gy'] = torch.symeig(ggt, eigenvectors=True)
        state['m2'] = Eg.unsqueeze(1) * Ex.unsqueeze(0) * state['num_locations']
        if group['layer_type'] == 'Conv2d' and self.sua:
            ws = group['params'][0].grad.data.size()
            state['m2'] = state['m2'].view(Eg.size(0), Ex.size(0), 1, 1).expand(-1, -1, ws[2], ws[3])

    def _get_gathering_filter(self, mod):
        """Convolution filter that extracts input patches."""
        kw, kh = mod.kernel_size
        g_filter = mod.weight.data.new(kw * kh * mod.in_channels, 1, kw, kh)
        g_filter.fill_(0)
        for i in range(mod.in_channels):
            for j in range(kw):
                for k in range(kh):
                    g_filter[k + kh*j + kw*kh*i, 0, j, k] = 1
        return g_filter

    def _to_kfe_sua(self, g, vx, vg):
        """Project g to the kfe"""
        sg = g.size()
        g = torch.mm(vg.t(), g.view(sg[0], -1)).view(vg.size(1), sg[1], sg[2], sg[3])
        g = torch.mm(g.permute(0, 2, 3, 1).contiguous().view(-1, sg[1]), vx)
        g = g.view(vg.size(1), sg[2], sg[3], vx.size(1)).permute(0, 3, 1, 2)
        return g

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()


def grad_wrt_kernel(a, g, padding, stride, target_size=None):
    gk = F.conv2d(a.transpose(0, 1), g.transpose(0, 1).contiguous(),
                  padding=padding, dilation=stride).transpose(0, 1)
    if target_size is not None and target_size != gk.size():
        return gk[:, :, :target_size[2], :target_size[3]].contiguous()
    return gk
