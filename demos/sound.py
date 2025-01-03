import argparse
from fft.stream_analyzer import Stream_Analyzer
import time
from key import Button
from subprocess import Popen
from xgolib import XGO
import threading
import inspect
import ctypes
import os,sys
dog = XGO(port='/dev/ttyAMA0',version="xgolite")
bt=Button()

def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=None, dest='device',
                        help='pyaudio (portaudio) device index')
    parser.add_argument('--height', type=int, default=240, dest='height',
                        help='height, in pixels, of the visualizer window')
    parser.add_argument('--n_frequency_bins', type=int, default=400, dest='frequency_bins',
                        help='The FFT features are grouped in bins')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--window_ratio', default='4/3', dest='window_ratio',
                        help='float ratio of the visualizer window. e.g. 24/9')
    parser.add_argument('--sleep_between_frames', dest='sleep_between_frames', action='store_true',
                        help='when true process sleeps between frames to reduce CPU usage (recommended for low update rates)')
    return parser.parse_args()

def convert_window_ratio(window_ratio):
    if '/' in window_ratio:
        dividend, divisor = window_ratio.split('/')
        try:
            float_ratio = float(dividend) / float(divisor)
        except:
            raise ValueError('window_ratio should be in the format: float/float')
        return float_ratio
    raise ValueError('window_ratio should be in the format: float/float')

exitcode = True
def checks():
    global exitcode
    time.sleep(6)
    while 1:
        dog.action(20)
        time.sleep(5)
        dog.action(22)
        time.sleep(5)
        dog.action(23)
        time.sleep(5)
    print("kill")
    dog.reset()

t = threading.Thread(target=checks)
t.start()



args = parse_args()
window_ratio = convert_window_ratio(args.window_ratio)

ear = Stream_Analyzer(
                device = args.device,        # Pyaudio (portaudio) device index, defaults to first mic input
                rate   = None,               # Audio samplerate, None uses the default source settings
                FFT_window_size_ms  = 60,    # Window size used for the FFT transform
                updates_per_second  = 100,   # How often to read the audio stream for new data
                smoothing_length_ms = 50,    # Apply some temporal smoothing to reduce noisy features
                n_frequency_bins = args.frequency_bins, # The FFT features are grouped in bins
                visualize = 1,               # Visualize the FFT features with PyGame
                verbose   = args.verbose,    # Print running statistics (latency, fps, ...)
                height    = args.height,     # Height, in pixels, of the visualizer window,
                window_ratio = window_ratio  # Float ratio of the visualizer window. e.g. 24/9
                )

fps = 60  #How often to update the FFT features + display
last_update = time.time()
proc=Popen("mplayer ./demos/fft/dream.mp3 -loop 0", shell=True)
while True:
    if bt.press_b():
        exitcode = False
        break
    raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()
print('main exit')
stop_thread(t)
dog.reset()
sys.exit()

