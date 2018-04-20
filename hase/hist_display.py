import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/andri/Documents/s2/5/master_arbeit/app")
import RLXYMSH

r=RLXYMSH()
hist=r.get_rlbwxh("chess.png")
print hist
