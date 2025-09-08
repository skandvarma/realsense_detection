#!/usr/bin/python
import pyrealsense2 as rs
import sys
import asyncio
import numpy as np
import pickle
import socket
import struct
import cv2

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
mc_ip_address = '224.0.0.1'
local_ip_address = '192.168.0.1'
port = 1024
chunk_size = 4096

class ImageClient:
    def __init__(self, reader, writer, source):   
        self.reader = reader
        self.writer = writer
        self.address = writer.get_extra_info('peername')[0]
        self.port = source[1]
        self.depth_buffer = bytearray()
        self.color_buffer = bytearray()
        self.windowName = self.port
        cv2.namedWindow("depth_window"+str(self.windowName))
        cv2.namedWindow("color_window"+str(self.windowName))
        self.depth_remaining = 0
        self.color_remaining = 0
        self.frame_id = 0
        self.state = 'header'  # 'header', 'depth', 'color'
       
    async def handle_connection(self):
        try:
            while True:
                if self.state == 'header':
                    # Read header: depth_size(4) + color_size(4) + timestamp(8)
                    header_data = await self.reader.readexactly(16)
                    self.depth_length = struct.unpack('<I', header_data[0:4])[0]
                    self.color_length = struct.unpack('<I', header_data[4:8])[0]
                    self.timestamp = struct.unpack('<d', header_data[8:16])[0]
                    
                    self.depth_remaining = self.depth_length
                    self.color_remaining = self.color_length
                    self.state = 'depth'
                
                elif self.state == 'depth':
                    # Read depth data
                    data = await self.reader.read(min(self.depth_remaining, chunk_size))
                    if not data:
                        break
                        
                    self.depth_buffer += data
                    self.depth_remaining -= len(data)
                    
                    if self.depth_remaining == 0:
                        self.state = 'color'
                
                elif self.state == 'color':
                    # Read color data
                    data = await self.reader.read(min(self.color_remaining, chunk_size))
                    if not data:
                        break
                        
                    self.color_buffer += data
                    self.color_remaining -= len(data)
                    
                    if self.color_remaining == 0:
                        self.handle_frame()
                        self.state = 'header'
                        
        except asyncio.IncompleteReadError:
            print(f"Connection closed for port {self.port}")
        except Exception as e:
            print(f"Error in image client: {e}")
        finally:
            cv2.destroyWindow("depth_window"+str(self.windowName))
            cv2.destroyWindow("color_window"+str(self.windowName))
            self.writer.close()
            await self.writer.wait_closed()
    
    def handle_frame(self):
        try:
            # Process depth frame
            depth_bytes = bytes(self.depth_buffer)
            depth_data = pickle.loads(depth_bytes, encoding='latin-1')
            
            if hasattr(depth_data, 'dtype') and depth_data.dtype != np.uint16:
                depth_data = depth_data.astype(np.uint16)
                
            bigDepth = cv2.resize(depth_data, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST) 
            cv2.putText(bigDepth, f"Depth {self.timestamp:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (65536), 2, cv2.LINE_AA)
            cv2.imshow("depth_window"+str(self.windowName), bigDepth)
            
            # Process color frame
            color_bytes = bytes(self.color_buffer)
            color_data = pickle.loads(color_bytes, encoding='latin-1')
            
            bigColor = cv2.resize(color_data, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cv2.putText(bigColor, f"Color {self.timestamp:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("color_window"+str(self.windowName), bigColor)
            
            cv2.waitKey(1)
            
            # Clear buffers for next frame
            self.depth_buffer = bytearray()
            self.color_buffer = bytearray()
            self.frame_id += 1
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            self.depth_buffer = bytearray()
            self.color_buffer = bytearray()

class EtherSenseClient:
    def __init__(self):
        self.server_address = ('', 1025)
        
    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        print('Incoming connection from %s' % repr(addr))
        
        # When a connection is attempted, delegate image receival to the ImageClient 
        handler = ImageClient(reader, writer, addr)
        await handler.handle_connection()
        
    async def start_server(self):
        server = await asyncio.start_server(
            self.handle_client, 
            self.server_address[0], 
            self.server_address[1]
        )
        
        addr = server.sockets[0].getsockname()
        print(f'Listening for connections on {addr}')
        
        async with server:
            await server.serve_forever()

async def multi_cast_message(ip_address, port, message):
    # Send the multicast message
    multicast_group = (ip_address, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        print('sending "%s" to %s' % (message, str(multicast_group)))
        sent = sock.sendto(message.encode(), multicast_group)
   
        # Start the client server to receive connections
        client = EtherSenseClient()
        await client.start_server()
        
    except socket.timeout:
        print('timed out, no more responses')
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print('closing socket')
        sock.close()

async def main():
    await multi_cast_message(mc_ip_address, port, 'EtherSensePing')

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Client stopped")
        cv2.destroyAllWindows()
