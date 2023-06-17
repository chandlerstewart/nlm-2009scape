import socket
import json
import struct
import constants
from QLearning import *
import torch
import threading
import signal
import traceback
from utils import Message, State, Bot
import utils






class Server:

    def __init__(self):
        self.MESSAGE_IN_UPDATED = False
        self.MESSAGE_IN = Message("temp")
        self.MESSAGE_OUT = Message("waiting_for_connection")
        self.MESSAGE_IN_UPDATED = False
        self.SOCKET_OPEN = True
        self.STATE = State.WAIT_FOR_CONNECTION


    def start(self):

        def server_handler(signum, frame):
            self.SOCKET_OPEN = False
            exit(1)

        signal.signal(signal.SIGINT, server_handler)
        server_thread = threading.Thread(target = self._start)
        server_thread.start()


    def _start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((constants.HOST, constants.PORT))
        server_socket.listen()


        while self.SOCKET_OPEN:
            client_socket, addr = server_socket.accept()
            length_bytes = client_socket.recv(2)

            if len(length_bytes) == 2:
                data_size = struct.unpack('>H', length_bytes)[0]

                #print(f"Connected by {addr}")
                data = client_socket.recv(data_size)
                while(len(data) < data_size):
                    chunk = client_socket.recv(data_size - len(data))
                    if not chunk:
                        break

                    data += chunk

            else:
                print("FIX THIS!")


            self.MESSAGE_IN = Message(**json.loads(data.decode('utf-8')))
            self.MESSAGE_IN_UPDATED = True
            self.manage_state(self.MESSAGE_IN)

            #print(self.MESSAGE_OUT.command)
            #print(self.MESSAGE_OUT.info)
            response = self.MESSAGE_OUT.to_json_out()
            
            
            client_socket.send(response)
            client_socket.close()


    def manage_state(self,message):
        self.last_response = message
        if self.STATE == State.RESET_EPISDOE:
            return
        elif message.command == "Connected":
            self.STATE = State.SPAWN_BOTS
            print("State: Spawn Bots")
        elif message.command in ["Wait", "Success: spawn_bots"]:
            self.STATE = State.WAIT_FOR_DATA
            print("State: Waiting for DATA")
        elif message.command == "json":
            self.last_response = json.loads(message.info)
            self.STATE = State.SEND_ACTION

    def step(self,data):
        self.MESSAGE_OUT = Message("json", data)
        
    
    def close(self):
        self.SOCKET_OPEN = False

        



    def update_message(self):
        if self.STATE == State.WAIT_FOR_CONNECTION:
            self.MESSAGE_OUT = Message("waiting_for_connection")
        if self.STATE == State.WAIT_FOR_DATA:
            self.MESSAGE_OUT = Message("server_waiting")
        if self.STATE in [State.SPAWN_BOTS, State.RESET_EPISDOE]:
            self.STATE = State.SPAWN_BOTS
            botinfo = Bot({"task":"woodcutting", "nodesRange": constants.NODES_RANGE})
            botinfo = json.dumps([botinfo.info])
            self.MESSAGE_OUT = Message(f"spawn_bots {constants.SPAWN_LOCATION[0]} {constants.SPAWN_LOCATION[1]} {constants.NUM_BOTS}", botinfo )


        self.MESSAGE_IN_UPDATED = False



 
        
    











 




#batch_size = 128
#epochs = 5
#lr = 0.0001


#AGENT = QLearningAgent(constants.STATE_SIZE, constants.ACTION_SIZE, lr=lr)
#REPLAY_MEMORY = ReplayMemory()
#EPISODE_NUM_STEPS = constants.EPISODE_NUM_STEPS_MIN
#STEPS_THIS_EPISODE = 0




    