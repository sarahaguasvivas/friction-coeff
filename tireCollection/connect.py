#! /usr/bin/env python2.7
import socket
import struct
import sys
import numpy as np
BUFFER_SIZE= 32000
filename= open('tireDataFakeGrass.txt', 'w')
sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('192.168.50.131', 5005))
print "connection with ESP established"
try:
    while True:
        data= sock.recv(BUFFER_SIZE)
        str1= str(len(data)/4) + "f"
        window= struct.unpack(str1, data)
        window1= np.array(window)
        np.savetxt(filename, window1, fmt= "%1.4f")
except Exception as e:
    filename.close()
    sock.close()
    print "socket closed"
    print str(e)

