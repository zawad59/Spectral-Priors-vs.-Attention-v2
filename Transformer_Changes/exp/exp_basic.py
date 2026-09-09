import os
import torch
from models import (
    Autoformer,
    Crossformer,
    FEDformer,
    Informer,
    iTransformer,
    MTST,
    Nonstationary_Transformer,
    PatchTST,
    Reformer,
    Transformer,
    TCN,
    Medformer,
    Transformer_2,
    Spectral_Transformer,
    Temporal_Transformer,
    Conformer, TeCh
)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "Autoformer": Autoformer,
            "Crossformer": Crossformer,
            "FEDformer": FEDformer,
            "Informer": Informer,
            "iTransformer": iTransformer,
            "MTST": MTST,
            "Nonstationary_Transformer": Nonstationary_Transformer,
            "PatchTST": PatchTST,
            "Reformer": Reformer,
            "Transformer": Transformer,
            "TCN": TCN,
            "Medformer": Medformer,
            "Transformer_2": Transformer_2,
            "Spectral_Transformer": Spectral_Transformer,
            "Temporal_Transformer": Temporal_Transformer,
            "Conformer": Conformer,
            "TeCh": TeCh
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
