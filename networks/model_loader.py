from .freqnet import FreqNet as FreqNet1
from .freqnet2 import FreqNet as FreqNet2
from .freqnet3 import FreqNet as FreqNet3

def get_model_by_name(name):
    if name == 'freqnet1':
        return FreqNet1()
    elif name == 'freqnet2':
        return FreqNet2()
    elif name == 'freqnet3':
        return FreqNet3()
    else:
        raise ValueError(f"Unknown model name: {name}")
