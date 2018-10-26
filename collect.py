#!/usr/bin/env python
import struct
import socket
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
BUFFER_SIZE=4*4*500
NUM_ADC=4

IP_1= '192.168.50.131'
filename= open(str(sys.argv[1])+'.txt',"w")
sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP_1, 5005))
print("Connection established!")
try:
	while (1):
		data= sock.recv(BUFFER_SIZE)
		filename.write(str(data))

except Exception as e:
	print(str(e))

