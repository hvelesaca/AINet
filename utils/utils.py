import torch
import numpy as np
from thop import profile
from thop import clever_format


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    """
    Ajusta el learning rate del optimizador basado en un esquema de decaimiento por pasos.

    Args:
        optimizer (torch.optim.Optimizer): El optimizador cuyos learning rates se ajustarán.
        init_lr (float): El learning rate inicial.
        epoch (int): El número de la época actual (empezando desde 1).
        decay_rate (float): El factor por el cual decae el learning rate (e.g., 0.1 para reducir 10x).
        decay_epoch (int): La frecuencia (en épocas) con la que ocurre el decaimiento.
    """
    # Calcula cuántas veces debería haber ocurrido el decaimiento hasta esta época
    # Nota: Si epoch=30, exponente=1. Si epoch=59, exponente=1. Si epoch=60, exponente=2.
    exponent = (epoch -1) // decay_epoch # Usar epoch-1 si las épocas empiezan en 1

    # Calcula el nuevo learning rate objetivo basado en el inicial y el decaimiento
    new_lr = init_lr * (decay_rate ** exponent)

    # Establece el nuevo learning rate en todos los grupos de parámetros del optimizador
    for param_group in optimizer.param_groups:
        # Es buena idea verificar si realmente cambió para evitar operaciones innecesarias
        # y facilitar el logging si solo quieres imprimir cuando cambia.
        if param_group['lr'] != new_lr:
            param_group['lr'] = new_lr
            # Opcional: Imprimir solo cuando cambia
            # print(f"Epoch {epoch}: Learning rate ajustado a {new_lr:.6f}")

def adjust_lr2(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
