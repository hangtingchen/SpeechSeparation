# -*- coding: utf-8 -*-
'''
@author : chenhangting 
@data : 2018/12/12
@note : Contains various transforms
        Part of tranforms are from 
        https://github.com/yongxuUSTC/sednn
'''
import numpy as np
from scipy import signal
import decimal
import argparse
import soundfile
import pprint

# znorm part
def znorm(x,mean,std):
    assert x.shape[-1]==mean.shape[-1] \
        and x.shape[-1]==std.shape[-1], \
        "Size not match for x {}, mean {}, std {}".format(x.shape,mean.shape,std.shape)
    return (x-mean)/std

def inv_znorm(x,mean,std):
    assert x.shape[-1]==mean.shape[-1] \
        and x.shape[-1]==std.shape[-1], \
        "Size not match for x {}, mean {}, std {}".format(x.shape,mean.shape,std.shape)
    return x*std+mean

# config part
class wavConfig():
    def __init__(self,sample_rate=16000,n_window=512,n_overlap=256):
        self.sample_rate = sample_rate
        self.n_window = n_window      # windows size for FFT
        self.n_overlap = n_overlap     # overlap of window

# calculate lps part
def calc_lps_wapper(audio, mode, cfg):
    assert mode=='magnitude',"Calculation of lps is based on magnitude"
    return log_sp(calc_sp(audio,mode,cfg))

def calc_lps_wapper2(audio, cfg):
    sp=calc_sp(audio,"complex",cfg)
    return log_sp(np.abs(sp)),np.angle(sp)

def calc_sp(audio, mode, cfg):
    """Calculate spectrogram. 
    
    Args:
      audio: 1darray. 
      mode: string, 'magnitude' | 'complex'
    
    Returns:
      spectrogram: 2darray, (n_time, n_freq). 
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    ham_win = np.hamming(n_window)
    # pad signal
    pad_width=(audio.shape[0]-n_window)%n_overlap;pad_width=n_overlap-pad_width if pad_width>0 else 0
    audio=np.pad(audio,pad_width=((0,pad_width),),mode='constant')
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode=mode) 
    x = x.T
    if mode == 'magnitude' or mode == 'angle':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x

def log_sp(x):
    return np.log(x + 1e-08)

# recover wave part
# TODO 找到原因
# 当你复原时，发现生成的音频和原始音频相比，在很多地方有+1的误差
# 虽然不会影响听感和直观的波形
def recover_lps2wav_wapper(lps,theta,cfg,norm=False):
    sp=np.exp(lps)
    s=recover_wav(sp,theta,cfg.n_overlap,np.hamming)
    s*=np.sqrt((np.hamming(cfg.n_window)**2).sum())
    if norm:
        s = s / np.max(np.abs(s))
    return s

def recover_wav(pd_abs_x, theta, n_overlap, winfunc, wav_len=None):
    """Recover wave from spectrogram. 
    If you are using scipy.signal.spectrogram, you may need to multipy a scaler
    to the recovered audio after using this function. For example, 
    recover_scaler = np.sqrt((ham_win**2).sum())
    
    Args:
      pd_abs_x: 2d array, (n_time, n_freq)
      theta: 2d array, (n_time, n_freq)
      n_overlap: integar. 
      winfunc: func, the analysis window to apply to each frame.
      wav_len: integer. Pad or trunc to wav_len with zero. 
      
    Returns:
      1d array. 
    """
    assert pd_abs_x.shape==theta.shape,"Size mismatch {} and {}".format(pd_abs_x.shape,theta.shape)
    x = real_to_complex(pd_abs_x, theta)
    x = half_to_whole(x)
    frames = ifft_to_wav(x)
    (n_frames, n_window) = frames.shape
    s = deframesig(frames=frames, siglen=0, frame_len=n_window, 
                   frame_step=n_window-n_overlap, winfunc=winfunc)
    if wav_len:
        s = pad_or_trunc(s, wav_len)
    return s

def real_to_complex(pd_abs_x, theta):
    """Recover pred spectrogram's phase from ground truth's phase. 
    
    Args:
      pd_abs_x: 2d array, (n_time, n_freq)
      theta: 2d array, (n_time, n_freq)
      
    Returns:
      2d complex array, (n_time, n_freq)
    """
    cmplx = pd_abs_x * np.exp(1j * theta)
    return cmplx

def half_to_whole(x):
    """Recover whole spectrogram from half spectrogram. 
    """
    return np.concatenate((x, np.fliplr(np.conj(x[:, 1:-1]))), axis=1)

def ifft_to_wav(x):
    """Recover wav from whole spectrogram"""
    return np.real(np.fft.ifft(x))

def deframesig(frames,siglen,frame_len,frame_step,winfunc=lambda x:np.ones((x,))):    
    """Does overlap-add procedure to undo the action of framesig.
    Ref: From https://github.com/jameslyons/python_speech_features
    
    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'
 
    indices = np.tile(np.arange(0,frame_len),(numframes,1)) + np.tile(np.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = np.array(indices,dtype=np.int32)
    padlen = (numframes-1)*frame_step + frame_len   
    
    if siglen <= 0: siglen = padlen
    
    rec_signal = np.zeros((padlen,))
    window_correction = np.zeros((padlen,))
    win = winfunc(frame_len)
    
    for i in range(0,numframes):
        window_correction[indices[i,:]] = window_correction[indices[i,:]] + win + 1e-15 #add a little bit so it is never zero
        rec_signal[indices[i,:]] = rec_signal[indices[i,:]] + frames[i,:]
        
    rec_signal = rec_signal/window_correction
    return rec_signal[0:siglen]

def pad_or_trunc(s, wav_len):
    if len(s) >= wav_len:
        s = s[0 : wav_len]
    else:
        s = np.concatenate((s, np.zeros(wav_len - len(s))))
    return s

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

# For debug
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav',type=str,required=True,help='Wave file path')
    parser.add_argument('--fs',type=int,required=False,default=16000,help='sample rate')
    parser.add_argument('--window',type=int,required=False,default=512,help='fft window size')
    parser.add_argument('--overlap',type=int,required=False,default=256,help='overlap size')

    subparsers=parser.add_subparsers(dest='mode',help='Mode = wav2feat/feat2wav')

    parser1=subparsers.add_parser("wav2feat")
    parser1.add_argument('--feat',type=str,required=True,help='feat path')
    parser1.add_argument('--save_type',type=str,required=True,help='magnitude/complex/angle\n\
                        The magnitude is log of amplitude of spectral(LPS)\n\
                        The complex is spectral(sp)\n\
                        The angle is the angle of complex')

    parser2=subparsers.add_parser('feat2wav')
    parser2.add_argument('--lps',type=str,required=True,help='Lps feat path')
    parser2.add_argument('--angle',type=str,required=True,help='angle path')


    args=parser.parse_args()
    pprint.pprint(args.__dict__)
    cfg=wavConfig(args.fs,args.window,args.overlap)
    if(args.mode=="wav2feat"):
        (aud,fs)=soundfile.read(args.wav)
        assert fs==cfg.sample_rate, "Unmatched samplerate {}".format(fs)
        if(args.save_type=="magnitude"):
            lps=calc_lps_wapper(aud,'magnitude',cfg)
        elif(args.save_type=='complex' or args.save_type=='angle'):
            lps=calc_sp(aud,args.save_type,cfg)
        else:
            raise ValueError("Unexpted save_type {}".format(args.save_type))
        np.save(args.feat,lps)
    elif(args.mode=="feat2wav"):
        lps=np.load(args.lps);angle=np.load(args.angle)
        if(np.any(np.iscomplex(lps)) or np.any(np.iscomplex(angle))):
            raise ValueError("The lps and angle should be in feal field")
        aud=recover_lps2wav_wapper(lps,angle,cfg)
        soundfile.write(file=args.wav,data=aud,samplerate=cfg.sample_rate)
    else:
        raise Exception("Unexpected mode {}".format(args.mode))
        