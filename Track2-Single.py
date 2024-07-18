import numpy as np
import pandas as pd
import keras
import ruptures as rpt
from collections import Counter


def read_fovs(nfov, dir_data):
  fovs_trajs = []
  fovs_trajs_2 = []
  fovs_rang_trajs = []
  fovs_ind = {}
  fovs_id = {}
  for fov in range(nfov):
    df = pd.read_csv(dir_data+f"trajs_fov_{fov}.csv")
    input = np.zeros((64,208, 2))
    rang_trajs = []
    grouped = df.groupby(['traj_idx'])

    for _,group in grouped:
        part = int(_[0])
        org, dest = int(group["frame"].iloc[0]), int(group["frame"].iloc[-1])
        rang_trajs.append((org, dest))
        input[part%64, org:dest+1, 0] = group["x"]
        input[part%64, org:dest+1, 1] = group["y"]

        if ((part+1)%64) == 0 and part != 0:

          max_value = np.amax(input[:,:,1])
          if max_value > 128:
            input[:,:,1][input[:,:,1] != 0]  = input[:,:,1][input[:,:,1] != 0]-(max_value-128)
          max_value = np.amax(input[:,:,0])
          if max_value > 128:
            input[:,:,0][input[:,:,0] != 0]  = input[:,:,0][input[:,:,0] != 0]-(max_value-128)

          for indindex in range(1,208):
            input[:,-indindex] = input[:,-indindex]-input[:,-indindex-1]

          input_copy = input.copy()
          input[:,0] = np.zeros((64, 2))
          fovs_trajs.append(input)
          fovs_trajs_2.append(input_copy)
          fovs_rang_trajs.append(rang_trajs)
          fovs_id[len(fovs_rang_trajs)-1] = fov
          fovs_ind[len(fovs_rang_trajs)-1] = (((part+1)-64)//64)*64
          input = np.zeros((64,208, 2))
          rang_trajs = []

    if len(rang_trajs) > 0:
      max_value = np.amax(input[:,:,1])
      if max_value > 128:
        input[:,:,1][input[:,:,1] != 0]  = input[:,:,1][input[:,:,1] != 0]-(max_value-128)
      max_value = np.amax(input[:,:,0])
      if max_value > 128:
        input[:,:,0][input[:,:,0] != 0]  = input[:,:,0][input[:,:,0] != 0]-(max_value-128)
      for indindex in range(1,208):
        input[:,-indindex] = input[:,-indindex]-input[:,-indindex-1]
      input_copy = input.copy()
      input[:,0] = np.zeros((64, 2))
      fovs_trajs.append(input)
      fovs_trajs_2.append(input_copy)
      fovs_rang_trajs.append(rang_trajs)
      fovs_id[len(fovs_rang_trajs)-1] = fov
      fovs_ind[len(fovs_rang_trajs)-1] = (abs((part+1)-64)//64)*64
      input = np.zeros((64,208, 2))
      rang_trajs = []

  print(np.amax(np.array(fovs_trajs)))

  return fovs_trajs, fovs_trajs_2, fovs_rang_trajs, fovs_id, fovs_ind

def normalizar_pred(signal):
  for i in range(1, len(signal)-2):
    if signal[i-1] != signal[i] and (signal[i+1] != signal[i] or signal[i+2] != signal[i]) :
      signal[i] = signal[i-1]
  return signal

def change_points_model(pred_states, org, dest):
  signal = normalizar_pred(pred_states[org:dest])
  model = "l2"  # Model used for segmentation
  algo = rpt.Window(width = 10, model=model, jump = 1, min_size = 3).fit(signal)
  cp = algo.predict(pen=1)
  return cp


def change_pointsold(pred_coef, org, dest):
  try:
    signal = pred_coef[org:dest]
    model = "l2"  # Model used for segmentation
    algo = rpt.Window(width = 12, model=model, jump = 1, min_size = 3).fit(signal)
    cp = algo.predict(pen=29)
    return cp
  except:
    return []

def change_pointslog(pred_coef, org, dest):
  signal = pred_coef[org:dest]
  model = "l2"  # Model used for segmentation
  algo = rpt.Window(width = 18, model=model, jump = 1, min_size = 3).fit(signal)
  cp = algo.predict(pen = 5)
  for c in cp[:-1]:
    c -= 1
  return cp

def change_pointsnolog(pred_coef, org, dest):
  signal = pred_coef[org:dest]
  model = "l2"  # Model used for segmentation
  algo = rpt.Window(width = 18, model=model, jump = 1, min_size = 3).fit(signal)
  cp = algo.predict(pen = 5)
  return cp


def pred_trajs_fov(nfov, dir_pred, dir_data, unet, unet_alpha, unet_ks, unet_states):
  """
  Explain function
  """

  fovs_trajs,fovs_trajs2, fovs_rang_trajs, fovs_id, fovs_ind = read_fovs(nfov, dir_data)
  pred_alphas = unet_alpha.predict(np.array(fovs_trajs2), verbose=0)
  pred_ks = unet_ks.predict(np.array(fovs_trajs2), verbose=0)
  pred_trajs = np.concatenate((pred_ks, pred_alphas), axis=-1)
  pred_unet = unet.predict(np.array(fovs_trajs), verbose=0)
  pred_states = np.argmax(unet_states.predict(np.array(fovs_trajs2), verbose=0), axis=-1)

  pred_ks_log = pred_ks.copy()
  pred_ks_log[pred_ks_log == 0] = 10e-4
  pred_logk = np.log(pred_ks_log)
  pred_trajs_log = np.concatenate((pred_logk, pred_alphas), axis=-1)

  preb = -1

  states_stats = np.unique(pred_states, return_counts=True)
  ch_mode = 0

  if len(states_stats[1]) == 3:
    if states_stats[1][1] > 1000:
      ch_mode = 1

  for img, fov in fovs_id.items():
    submission_file = dir_pred + f'/fov_{fov}.txt'
    if preb == fov:
      val_text = "a"
    else:
      val_text = "w"

    preb = fov
    add_ind = fovs_ind[img]
    with open(submission_file, val_text) as f:
      for id_part, ran in enumerate(fovs_rang_trajs[img]):
        if ch_mode==0:
            dataK = pred_ks[img, id_part][ran[0]:ran[1]].flatten()
            Q2 = np.percentile(dataK, 50)
            if Q2 < 0.2:
              cp_part = change_pointsold(pred_unet[img, id_part], ran[0], ran[1])
            elif Q2 > 3:
              cp_part = change_pointslog(pred_trajs_log[img, id_part], ran[0], ran[1])
            else:
              cp_part = change_pointsnolog(pred_trajs[img, id_part], ran[0], ran[1])
        else:
            cp_part = change_points_model(pred_states[img, id_part], ran[0], ran[1])

        if cp_part[-1] == ran[1]-ran[0]:
          cp_part= cp_part[:-1]

        if len(cp_part) == 0:
          alpha = np.median(pred_alphas[img,id_part, ran[0]:ran[1]])
          ks = np.median(pred_unet[img,id_part, ran[0]:ran[1], 1])
          state = Counter(pred_states[img,id_part, ran[0]:ran[1]]).most_common(1)[0][0]
          state =  state-1

          if ch_mode==1:
              state = Counter(pred_states[img,id_part, ran[0]:cp_part[0]+ran[0]]).most_common(1)[0][0]
              state =  state-1
              if state == 1 and alpha > 1.7:
                state= 3
          else:
            state = 2

          res = [int(id_part+add_ind), ks, alpha, int(state), int(ran[1]-ran[0]+1)]
          formatted_numbers = ','.join(map(str, res))
          f.write(formatted_numbers + '\n')
        else:
          alpha = np.median(pred_alphas[img,id_part, ran[0]:cp_part[0]+ran[0]])
          ks = np.median(pred_unet[img,id_part, ran[0]:cp_part[0]+ran[0],1])
          state = Counter(pred_states[img,id_part, ran[0]:ran[1]]).most_common(1)[0][0]
          state =  state-1
          if ch_mode==1:
              state = Counter(pred_states[img,id_part, ran[0]:cp_part[0]+ran[0]]).most_common(1)[0][0]
              state =  state-1
              if state == 1 and alpha > 1.7:
                state= 3
          else:
            state = 2

          res = [int(id_part+add_ind), ks, alpha, state, int(cp_part[0]+1)]
          cp_part = cp_part+[int(ran[1]-ran[0])]
          for i in range(len(cp_part)-1):
            alpha = np.median(pred_alphas[img,id_part,int(cp_part[i]+ran[0]):int(cp_part[i+1]+ran[0])])
            ks = np.median(pred_unet[img,id_part,int(cp_part[i]+ran[0]):int(cp_part[i+1]+ran[0]), 1])
            if ch_mode==1:
                state = Counter(pred_states[img,id_part, ran[0]:cp_part[0]+ran[0]]).most_common(1)[0][0]
                state =  state-1
                if state == 1 and alpha > 1.7:
                    state= 3
            else:
                state = 2

            res.extend([ks, alpha, state, int(cp_part[i+1]+1)])
          formatted_numbers = ','.join(map(str, res))
          f.write(formatted_numbers + '\n')

unet = keras.models.load_model("/content/drive/My Drive/unet_v1.1.keras", compile=False)
unet_alpha = keras.models.load_model("/content/drive/MyDrive/att_unet/alphas-3-2-1024-v5-epoch-2.keras", compile=False)
unet_ks = keras.models.load_model(f"/content/drive/MyDrive/att_unet/ks-3-2-512-v3-epoch-1.keras", compile=False)
unet_states = keras.models.load_model(f"/content/drive/MyDrive/att_unet/states-3-6-128-v1-epoch-10.keras", compile=False)

experiments = 11 #Introduce the number of experiments
no_fovs = 30 #Introduce the number of fovs

for exp in range(int(experiments)):
  print("Pred. file ", exp)
  dir_data = rf"/content/drive/MyDrive/public_data_challenge_v0/track_2/exp_{exp}/" #Files directory
  dir_pred = rf"/content/drive/MyDrive/Model_015/track_2/exp_{exp}" #Predictions directory
  pred_trajs_fov(int(no_fovs), dir_pred, dir_data, unet_alpha, unet_ks, unet_states)



