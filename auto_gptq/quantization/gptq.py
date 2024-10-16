import math
import os
import time
from logging import getLogger

import torch
import torch.nn as nn
import transformers

from auto_gptq.quantization.quantizer import Quantizer


logger = getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.H = {
            "pos": torch.zeros((self.columns, self.columns), device=self.dev),
            "neg": torch.zeros((self.columns, self.columns), device=self.dev)
        }
        self.nsamples = {
            "pos": 0,
            "neg": 0,
        }
        self.quantizer = Quantizer()

        # Flag used in determining if sample is positive/negative
        # NOTE: We want good reconstruction on a positive batch and poor on negative
        self.is_positive_batch = True


    def add_batch(self, inp, out):
        """
        Hook to get input/output of each layer, and iteratively approximate the
        Hessian matrix.

        Parameters
        ----------
        inp : torch.Tensor
            Layer input
        out : torch.Tensor
            Layer output
        """
        if os.environ.get("DEBUG"):
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        # Hessian dampening factor
        key = "pos" if self.is_positive_batch else "neg"
        self.H[key] *= self.nsamples[key] / (self.nsamples[key] + tmp)
        self.nsamples[key] += tmp

        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples[key]) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())

        # Update moving average of Hessian
        self.H[key] += inp.matmul(inp.t())


    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        group_size=-1,
        actorder=False,
        static_groups=False,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H["pos"]) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + group_size)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        # NOTE: This isn't tested
        if actorder:
            raise NotImplementedError("Not tested")
            perm = torch.argsort(torch.diag(H["pos"]), descending=True)
            W = W[:, perm]
            H["pos"] = H["pos"][perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        # Dampen Hessian for positive samples with Hessian for negative samples
        H_pos = H["pos"]
        if self.nsamples["neg"] > 0:
            # Compute unit-norm eigenvectors for Hessian of negative sample
            lambda_neg, v_neg = torch.linalg.eigh(H["neg"])

            # Project positive sample Hessian onto the negative sample eigenvectors
            # to obtain a score (consider each projection independently)
            scores_pos = (v_neg.T @ H["pos"] @ v_neg).diag()

            # The equivalent projection of the negative sample Hessian results in
            # the eigenvalues
            scores_neg = lambda_neg

            # Compute dampening factor based on difference between projection scores
            # NOTE: Directions where negative samples Hessian > positive samples Hessian
            #       should be dampened. And directions where positive samples
            #       Hessian > negative samples Hessian can also be strengthened
            scores_diff = (scores_neg - scores_pos)
            damp = 1 - (scores_diff / scores_diff.max())
            # NOTE: Hessian needs to be positive-definite, so ensure damping factor
            #       is at least 0.001
            damp = torch.clamp(damp, min=0.001)

            # Reconstructing positive sample Hessian matrix with dampened scores
            H_pos = v_neg @ torch.diag(damp*scores_pos) @ v_neg.T

        # (Dampen) Add constant to diagonal to ensure positive-definite to
        #          ensure Cholesky Decomposition works fine
        damp = percdamp * torch.mean(torch.diag(H_pos))
        diag = torch.arange(self.columns, device=self.dev)
        H_pos[diag, diag] += damp

        # Compute inverse of Hessian (with Cholesky decomposition)
        H_lower = torch.linalg.cholesky(H_pos)
        H_inv = torch.cholesky_inverse(H_lower)

        # Compute Cholesky decomposition on inverse of Hessian
        H_inv_upper = torch.linalg.cholesky(H_inv, upper=True)
        Hinv = H_inv_upper

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:
                            self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + group_size)], weight=True)

                        if ((i1 + i) // group_size) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // group_size]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if os.environ.get("DEBUG"):
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                logger.debug(torch.sum(Losses))

        torch.cuda.synchronize()
        logger.info(f"duration: {(time.time() - tick)}")
        logger.info(f"avg loss: {torch.sum(Losses).item() / self.nsamples['pos']}")

        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)
        if os.environ.get("DEBUG"):
            logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx

    def free(self):
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


__all__ = ["GPTQ"]
