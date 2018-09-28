#!/usr/bin/env python
import struct
import socket
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

filename= str(sys.argv[1]) + '.csv'
print('Writing to ' + filename)

BUFFER_SIZE=  32000
sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('192.168.50.131', 5005))
print("Connection established!")

try:
	while (1):
		data= sock.recv(BUFFER_SIZE)
		str1= str(len(data)/4)+ "f"
		window= struct.unpack(str1, data)
		windoww= window[:-1-(len(window) % 4)+1]
		windoww= np.reshape(windoww, (-1, 4))
		window= pd.DataFrame(windoww)
		window.to_csv(filename, mode='a', header= False)

except Exception as e:
	print('Socket Closed')
	print(str(e))
	sock.close()
