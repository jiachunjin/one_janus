from .decoder_diffusion import Decoder_diffusion
from .decoder_ff import Decoder_ff
from .decoder_2d import Decoder_2D, Decoder_1D
from .sem_decoder_reg import Sem_Decoder_Reg
from .sem_decoder_woreg import Sem_Decoder_without_Reg

def get_decoder(config):
    if config.type == "diffusion":
        decoder = Decoder_diffusion(config)
    elif config.type == "ff":
        decoder = Decoder_ff(config)
    elif config.type == "sem_reg":
        decoder = Sem_Decoder_Reg(config)
    elif config.type == "sem_woreg":
        decoder = Sem_Decoder_without_Reg(config)
    elif config.type is None:
        decoder = None
    
    return decoder