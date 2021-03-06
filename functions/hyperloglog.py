import numpy as np
import pandas as pd
import random

# Question 1: HyperLogLog

# 1.1 Hash function
class HashLogLog:
    """Implementation of family of universal hashing functions.
    The function get_hash returns the binary value for
    ax + b mod p mod m"""
    def __init__(self, bits=32):
        random.seed(42)
        self.bits = bits
        self.m = 2 ** self.bits
        self.p = (2**148 + 1) // 17 # Prime number
        self.a = random.randint(1, self.p)
        self.b = random.randint(0, self.p)
        assert(self.m < self.p), "Too many bits."
    
    def get_hash(self, x):
        x = int(x, 16)
        bin_hash = bin(((self.a * x + self.b) % self.p) % self.m)[2:]
        return bin_hash.zfill(self.bits)

# 1.2 HyperLogLog data structure
class HyperLogLog:
    """Implementation of the HyperLogLog structure based on the work by Flajolet et al."""
    def __init__(self, log2m, bits=32):
        self.log2m = log2m
        self.bits = bits
        self.my_hash = HashLogLog(bits=self.bits)
        self.m = 2 ** self.log2m
        assert(self.m in [16, 32, 64] or self.m >= 128)
    
    def structure(self, file_path):
        """Returns the HyperLogLog structure for a text file containing exadecimal strings.

        Args:
            file_path (str): Path for the file containing the strings to be analyzed.

        Returns:
            list: List representing the HyperLogLog structure.
        """
        HLL = [0] * self.m
        with open(file_path) as f:
            for line in f:
                hashed = self.my_hash.get_hash(line)
                root = int(hashed[:self.log2m], 2)
                try:
                    temp = hashed[self.log2m:].index('1') + 1
                except:
                    temp = len(hashed[self.log2m:])
                if temp > HLL[root]:
                    HLL[root] = temp
        return HLL
        
    def cardinality(self, hll):
        """Even if there is a formula to compute alpha, the values are directly
        taken from the paper as the are for common bits numbers."""
        d = {16: 0.673, 32: 0.697, 64: 0.709}
        if self.m >= 128:
            alpha = 0.7213 / (1 + 1.079 / self.m)
        else:
            alpha = d[self.m]
        
        return int(self.m ** 2 * alpha * (1 / sum([2**(-x) for x in hll])))
    
    def error(self):
        return 1.04 / np.sqrt(self.m)