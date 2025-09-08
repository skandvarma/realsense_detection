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
port = 1024
chunk_size = 4096

def getDepthAndTimestamp(pipeline, depth_filter):
    frames = pipeline.wait_for_frames()
    frames.keep()
    depth = frames.get_depth_frame()
    if depth:
        depth2 = depth_filter.process(depth)
        depth2.keep()
        depthData = depth2.as_frame().get_data()        
        depthMat = np.asanyarray(depthData)
        ts = frames.get_timestamp()
        return depthMat, ts
    else:
        return None, None

def openPipeline():
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline = rs.pipeline()
    pipeline_profile = pipeline.start(cfg)
    sensor = pipeline_profile.get_device().first_depth_sensor()
    return pipeline

class EtherSenseServer:
    def __init__(self, address):
        print("Launching Realsense Camera Server")
        try:
            self.pipeline = openPipeline()
        except:
            print("Unexpected error: ", sys.exc_info()[1])
            sys.exit(1)
        
        self.decimate_filter = rs.decimation_filter()
        self.decimate_filter.set_option(rs.option.filter_magnitude, 2)
        self.address = address
        self.packet_id = 0        

    async def handle_client(self, reader, writer):
        print("Connection received from", writer.get_extra_info('peername'))
        
        try:
            while True:
                frame_data = self.get_frame_data()
                if frame_data:
                    writer.write(frame_data)
                    await writer.drain()
                await asyncio.sleep(0.033)  # ~30 FPS
                
        except ConnectionResetError:
            print("Client disconnected")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    def get_frame_data(self):
        depth, timestamp = getDepthAndTimestamp(self.pipeline, self.decimate_filter)
        if depth is not None:
            # Use protocol 2 for compatibility
            data = pickle.dumps(depth, protocol=2)
            length = struct.pack('<I', len(data))
            ts = struct.pack('<d', timestamp)
            return length + ts + data
        return None

    async def start_server(self):
        server = await asyncio.start_server(
            self.handle_client, 
            self.address[0], 
            1025
        )
        
        addr = server.sockets[0].getsockname()
        print(f'Serving on {addr}')
        
        async with server:
            await server.serve_forever()

class MulticastServer:
    def __init__(self, host=mc_ip_address, port=1024):
        self.host = host
        self.port = port
        
    async def listen_for_multicast(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.port))
        sock.setblocking(False)
        
        print(f"Listening for multicast on port {self.port}")
        
        while True:
            try:
                data, addr = await asyncio.get_event_loop().sock_recvfrom(sock, 42)
                print('Received multicast message %s bytes from %s' % (len(data), addr))
                
                # Start EtherSense server for this client
                ethersense = EtherSenseServer(addr)
                asyncio.create_task(ethersense.start_server())
                
            except Exception as e:
                print(f"Multicast error: {e}")
                await asyncio.sleep(0.1)

async def main():
    server = MulticastServer()
    await server.listen_for_multicast()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped")
