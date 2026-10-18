"""Optional camera read; no frame written, displayed or uploaded."""
import sys

def main():
    import cv2
    cap=cv2.VideoCapture(int(sys.argv[1]),cv2.CAP_V4L2)
    try:
        ok,frame=cap.read()
        if not ok or frame is None: raise RuntimeError('Camera read failed')
        print('Frame captured and discarded; width=%d height=%d'%(frame.shape[1],frame.shape[0]))
    finally: cap.release()
if __name__=='__main__': main()
