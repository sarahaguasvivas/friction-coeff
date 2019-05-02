#!/usr/bin/env python
import struct
import socket
import sys
import os

IP_5= '192.168.1.5'

BUFFER_SIZE= 32000

filename= os.path.join('data', str(sys.argv[1])+'.csv')
sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP_5, 5005))
print("Connection established!")
try:
    listl=[]
    while (1):
        data= sock.recv(BUFFER_SIZE)
        str1= str(len(data)/4) + "f"
        data= struct.unpack(str1, data)
        listl+= list(data)

except KeyboardInterrupt:
    print("saving data file")
    filef= open(filename, 'w')
    listl= ",".join(str(bit) for bit in listl)
    filef.write(listl)
