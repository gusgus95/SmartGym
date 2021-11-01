import camera
import data_transfer
from multiprocessing import shared_memory, Process, Semaphore
import numpy as np

# try catch 문 추가하기
# signal handling 추가하기

# ENVIRONMENT VARIABLE
numOfCamera = 1            # 사용할 카메라의 개수
numOfMachines = 5         # 인식할 기구의 개수

# this is a central process
if __name__ == "__main__":
    listOfSem = []          # 세마포어 객체들을 담는 리스트
    listOfCam = []          # 카메라 프로세스 객체들을 담는 리스트
    listOfMemBlock = []     # 공유 메모리 객체들을 담는 리스트

    arr = np.full(numOfMachines, True, dtype=bool)

    for seq in range(numOfCamera):
        listOfMemBlock.append(shared_memory.SharedMemory(create=True, size=arr.nbytes))
        listOfSem.append(Semaphore())
        listOfCam.append(Process(target=camera.get_bounding_box_of_human,
                                 args=(seq, str(seq), listOfMemBlock[seq].name, arr.shape, arr.dtype, listOfSem[seq])))

    transferProcess = Process(target=data_transfer.flask_transfer, args=([x.name for x in listOfMemBlock], arr.shape,
                                                                         arr.dtype, listOfSem))
    transferProcess.start()
    for cameraProcess in listOfCam:
        cameraProcess.start()

    transferProcess.join()
    for cameraProcess in listOfCam:
        cameraProcess.join()

    for shared in listOfMemBlock:
        shared.close()
        shared.unlink()
