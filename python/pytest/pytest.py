
from itertools import islice
from random import random
from time import perf_counter
import pyLonginus


if __name__ == "__main__":
    
    ins = pyLonginus.new_instance((-1))
    print(pyLonginus.get_version())