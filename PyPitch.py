#!/usr/bin/python3

import sounddevice as sd
import numpy as np
import scipy.fftpack
import os
import copy
from dearpygui.core import *
from dearpygui.simple import *
from math import cos
from random import random

# General settings
SAMPLE_FREQ = 48000 # sample frequency in Hz
WINDOW_SIZE = 48000 # window size of the DFT in samples
WINDOW_STEP = 12000 # step size of window
WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ # length between two samples in seconds
NUM_HPS = 8 #max number of harmonic product spectrums
DELTA_FREQ = (SAMPLE_FREQ/WINDOW_SIZE) # frequency step width of the interpolated DFT
windowSamples = [0 for _ in range(WINDOW_SIZE)]
noteBuffer = ["1","2","3"]

# This function finds the closest note for a given pitch
# Returns: note (e.g. a, g#, ..), pitch of the tone
CONCERT_PITCH = 440
ALL_NOTES = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]
def find_closest_note(pitch):
  i = int( np.round( np.log2( pitch/CONCERT_PITCH )*12 ) )
  clostestNote = ALL_NOTES[i%12] + str(4 + np.sign(i) * int( (9+abs(i))/12 ) )
  closestPitch = CONCERT_PITCH*2**(i/12)
  return clostestNote, closestPitch

hannWindow = np.hanning(WINDOW_SIZE)
currFrequency = 440
def callbackHPS(indata, frames, time, status):
  global windowSamples, lastNote, currFrequency
  if status:
    print(status)
  if any(indata):
    windowSamples = np.concatenate((windowSamples,indata[:, 0])) # append new samples
    windowSamples = windowSamples[len(indata[:, 0]):] # remove old samples

    signalPower = (np.linalg.norm(windowSamples, ord=2)**2) / len(windowSamples)
    if signalPower < 5e-7:
      # os.system('cls' if os.name=='nt' else 'clear')
      print("Closest note: ...")
      return

    hannSamples = windowSamples * hannWindow
    magnitudeSpec = abs( scipy.fftpack.fft(hannSamples)[:len(hannSamples)//2] )

    #supress mains hum
    for i in range(int(62/DELTA_FREQ)):
      magnitudeSpec[i] = 0

    #Calculate average energy per frequency for the octave bands
    octaveBands = [50,100,200,400,800,1600,3200,6400,12800,25600]
    for j in range(len(octaveBands)-1):
      indStart = int(octaveBands[j]/DELTA_FREQ)
      indEnd = int(octaveBands[j+1]/DELTA_FREQ)
      indEnd = indEnd if len(magnitudeSpec) > indEnd else len(magnitudeSpec)
      avgEnergPerFreq = 1*(np.linalg.norm(magnitudeSpec[indStart:indEnd], ord=2)**2) / (indEnd-indStart)
      avgEnergPerFreq = avgEnergPerFreq**0.5
      for i in range(indStart, indEnd):
        magnitudeSpec[i] = magnitudeSpec[i] if magnitudeSpec[i] > avgEnergPerFreq else 0  #suppress white noise

    #Interpolate spectrum
    magSpecIpol = np.interp(np.arange(0, len(magnitudeSpec), 1/NUM_HPS), np.arange(0, len(magnitudeSpec)), magnitudeSpec)
    magSpecIpol = magSpecIpol / np.linalg.norm(magSpecIpol, ord=2) #normalize it

    hpsSpec = copy.deepcopy(magSpecIpol)

    for i in range(NUM_HPS):
      tmpHpsSpec = np.multiply(hpsSpec[:int(np.ceil(len(magSpecIpol)/(i+1)))], magSpecIpol[::(i+1)])
      if not any(tmpHpsSpec):
        break
      hpsSpec = tmpHpsSpec

    maxInd = np.argmax(hpsSpec)
    maxFreq = maxInd * (SAMPLE_FREQ/WINDOW_SIZE) / NUM_HPS

    closestNote, closestPitch = find_closest_note(maxFreq)
    maxFreq = round(maxFreq, 1)
    closestPitch = round(closestPitch, 1)

    noteBuffer.insert(0,closestNote) #note that this is a ringbuffer
    noteBuffer.pop()

    majorityVote = max(set(noteBuffer), key = noteBuffer.count)

    if noteBuffer.count(majorityVote) > 1:
      detectedNote = majorityVote
    else:
      return
    # os.system('cls' if os.name=='nt' else 'clear')
    currFrequency = maxFreq
    print(f"Closest note: {closestNote} {maxFreq}/{closestPitch}")
  else:
    print('no input')

# The sounddecive callback function
# Provides us with new data once WINDOW_STEP samples have been fetched
def callbackDFT(indata, frames, time, status):
  global windowSamples, currFrequency
  if status:
    print(status)
  if any(indata):
    windowSamples = np.concatenate((windowSamples,indata[:, 0])) # append new samples
    windowSamples = windowSamples[len(indata[:, 0]):] # remove old samples
    magnitudeSpec = abs( scipy.fftpack.fft(windowSamples)[:len(windowSamples)//2] )

    for i in range(int(62/(SAMPLE_FREQ/WINDOW_SIZE))):
      magnitudeSpec[i] = 0 #suppress mains hum

    maxInd = np.argmax(magnitudeSpec)
    maxFreq = maxInd * (SAMPLE_FREQ/WINDOW_SIZE)
    closestNote, closestPitch = find_closest_note(maxFreq)

    # os.system('cls' if os.name=='nt' else 'clear')
    currFrequency = maxFreq
    print(f"Closest note: {closestNote} {maxFreq:.1f}/{closestPitch:.1f}")
  else:
    print('no input')

def getDummyFrequency():
  return 440 + random() / 200.0

def getMicFrequency():
  global currFrequency
  return currFrequency

def plot_callback(sender, data):
    # keeping track of frames
    frame_count = get_data("frame_count")
    frame_count += 1
    add_data("frame_count", frame_count)

    # updating plot data
    plot_datax = get_data("plot_datax")
    plot_datay = get_data("plot_datay")
    if(len(plot_datax) > 500):
      plot_datax.pop(0)
      plot_datay.pop(0)

    plot_datax.append(3.14 * frame_count / 180)
    plot_datay.append(getMicFrequency())
    add_data("plot_datax", plot_datax)
    add_data("plot_datay", plot_datay)

    # plotting new data
    add_line_series("Plot", "Frequency", plot_datax, plot_datay, weight=2)

with window("Frequency", width=500, height=500):
    try:
      print("Starting HPS guitar tuner...")
      with sd.InputStream(channels=1, callback=callbackDFT,
        blocksize = WINDOW_STEP,
        samplerate = SAMPLE_FREQ):
        # TODO: Add static min and max bounds
        add_plot("Plot", height=-1, xaxis_time=True,
          yaxis_lock_max = True,
          yaxis_lock_min = True
          )
        add_data("plot_datax", [])
        add_data("plot_datay", [])
        add_data("frame_count", 0)
        set_render_callback(plot_callback)
        start_dearpygui()
    except Exception as e:
      print(str(e))
