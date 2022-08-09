#!/usr/bin/env python3
from scipy.io import wavfile
import pyroomacoustics as pra
import numpy as np
from optparse import OptionParser

def bss_code(input_file, output_file, numiter, segsize, fft_size, channels):
   # read multichannel wav file
   # audio.shape == (nsamples, nchannels)
   #fs, audio = wavfile.read("PR0001_ECxxx_LZ_20190928.wav")
   fs, audio = wavfile.read(input_file)
   #fs, audio = wavfile.read("LKC_S1_final.wav")


   # STFT analysis parameters
   #numiter=20
   #segsize=20
   #fft_size = 4096  # `fft_size / fs` should be ~RT60
   hop = fft_size // 2  # half-overlap
   win_a = pra.hann(fft_size)  # analysis window
   # optimal synthesis window
   win_s = pra.transform.compute_synthesis_window(win_a, hop)

   length=audio.shape[0]
   print(length)

   ##### Getting the first W

   for nchan in range(0, channels, 1):
      audio2=audio[0:(fs*segsize),:]

      # STFT
      X = pra.transform.analysis(audio2, fft_size, hop, win=win_a)
      print(X.shape)
#      if len(X.shape) <= 2:
#          X = np.expand_dims(X, 2)

      # Separation
      Y, W_init = pra.bss.fastmnmf(X, n_iter=numiter, mic_index=nchan)
      print (W_init.shape)
      print("Y_shape_after_separation1")

      # iSTFT (introduces an offset of `hop` samples)
      # y contains the time domain separated signals
      # y.shape == (new_nsamples, nchannels)
      y = pra.transform.synthesis(Y, fft_size, hop, win=win_s)
      #y_norm = pra.normalize(y)
      y_final = y[0:(fs*segsize),:]

      initial = fs*segsize
      #######
      for shift in range(fs*segsize*2,length,fs*segsize):
         audio2=audio[initial:shift,:]
         X = pra.transform.analysis(audio2, fft_size, hop, win=win_a)
         Y, W = pra.bss.fastmnmf(X, n_iter=numiter, W0=W_init, mic_index=nchan)
         print(W.shape)
         y = pra.transform.synthesis(Y, fft_size, hop, win=win_s)
         y_norm = pra.normalize(y)
         y_final = np.concatenate(( y_final, y[0:(fs*segsize),:] ), axis=0)
         #print("Y_final_inside_loop_BEFORE_edge")
         #print(y_final.shape)
         initial = shift
         W_init = W
         #print("initial_shift_next_iter")
         #print(shift)
         #shift = shift + fs*20
      shift = length
      audio2=audio[initial:shift,:]
      X = pra.transform.analysis(audio2, fft_size, hop, win=win_a)
      Y, W = pra.bss.fastmnmf(X, n_iter=numiter, W0=W_init, mic_index=nchan)
      print(W.shape)
      y = pra.transform.synthesis(Y, fft_size, hop, win=win_s)
      #y_norm = pra.normalize(y)
      y_final = np.concatenate(( y_final, y[0:(fs*segsize),:] ), axis=0)
      y_final2 = pra.normalize(y_final[:,nchan])
      if nchan==0:
         y_final3 = y_final2[None,:]
      else:
         y_final3=np.concatenate((y_final3,y_final2[None,:]), axis=0 )
      print("Y_final_shape")
      print(y_final.shape)
      print("Y_final2_shape")
      print(y_final2.shape)
      print("Y_final3_shape")
      print(y_final3.shape)
      print("original_audio_length")
      print(audio.shape)


   wavfile.write(output_file,fs,y_final3.T)

####
def main():
   usage = "%prog [options] input_file output_file"
   desc = "produces BSS using FastMNMF"
   version = "%prog 0.1"
   parser = OptionParser(usage=usage, description=desc, version=version)
   (opt, args) = parser.parse_args()
   #numiter=20
   #segsize=20
   #fft_size = 4096
   #channels =2
   if(len(args)!=6):
      parser.error("Six arguments expected in this order number of iterations, segsize, fft_size, channels")
   input_file, output_file, numiter, segsize, fft_size, channels = args
   print(args)
   print(output_file)

   # get test file of protocol
   protocol = bss_code(input_file, output_file, int(numiter), int(segsize), int(fft_size), int(channels) )


if __name__=="__main__":
   main()

