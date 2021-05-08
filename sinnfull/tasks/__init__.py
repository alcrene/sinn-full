import sinnfull
if not sinnfull._setup_was_run:
    sinnfull.setup('theano')
    
from .base import *
from .analysis import *
