from multiprocessing import shared_memory, Semaphore
import numpy as np
from main import numOfMachines
import socket

from flask import Flask, json

import logging

app = Flask(__name__)

global listOfMemName, arrShape, arrType, semList


def flask_transfer(names: list, arrshape: tuple, arrtype: property, sems: list):
    global listOfMemName, arrShape, arrType, semList
    listOfMemName, arrShape, arrType, semList = names, arrshape, arrtype, sems
    app.run(host='0.0.0.0', port=5000)


@app.route('/flask/machine')
def machine_data():
    global listOfMemName, arrShape, arrType, semList
    machines = {}
    for i in range(numOfMachines):
        machines[i] = True

    for seq, sem in enumerate(semList):
        sem.acquire()       # 세마포어 획득
        connect_shared = shared_memory.SharedMemory(name=listOfMemName[seq])      # 일시적으로 메모리에 대한 접근권 획득
        # 로컬변수에 공유메모리데이터 저장
        temp_arr = np.ndarray(shape=arrShape, dtype=arrType, buffer=connect_shared.buf)
        logging.info(str(temp_arr))
        sem.release()       # 세마포어 반환
        logging.info(str(temp_arr))
        temp_list = temp_arr.tolist()
        for sequence, availability in enumerate(temp_list):
            machines[sequence] = machines[sequence] and availability
    return json.jsonify(machines)


# udp using flask server (Deprecated)
def udp_transfer(memory_name, shared_array, sem: Semaphore):
    host_ip = ''
    # socket server 포트는 5051 flask server 포트는 5050
    port = 5051
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.bind((host_ip, port))
    prev = {}  #
    while True:
        client_receive, client_addr = client_socket.recvfrom(1024)
        sem.acquire()
        new_sem = shared_memory.SharedMemory(name=memory_name)
        temp_arr = np.ndarray(shared_array.shape, dtype=shared_array.dtype, buffer=new_sem.buf).copy()
        text = str(temp_arr)
        sem.release()
        print(text)
        client_socket.sendto(text.encode('ascii'), client_addr)




