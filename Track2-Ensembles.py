import numpy as np
import pandas as pd
import keras
import ruptures as rpt
from collections import Counter
from sklearn.cluster import KMeans



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

def k_means_clusters(lista1, lista2):
    data = list(zip(lista1, lista2))

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(data)

    labels = kmeans.labels_

    clusters = [[[],[]],[[],[]]]
    for i, label in enumerate(labels):
        clusters[label][0].append(lista1[i])
        clusters[label][1].append(lista2[i])
    return clusters


def pred_trajs_fov(nfov, dir_pred, dir_data, unet_alpha, unet_ks, unet_states):
  fovs_trajs,fovs_trajs2, fovs_rang_trajs, fovs_id, fovs_ind = read_fovs(nfov, dir_data)

  pred_alphas = unet_alpha.predict(np.array(fovs_trajs2), verbose=0)
  pred_ks = unet_ks.predict(np.array(fovs_trajs2), verbose=0)
  pred_trajs = np.concatenate((pred_ks, pred_alphas), axis=-1)
  pred_states = np.argmax(unet_states.predict(np.array(fovs_trajs2), verbose=0), axis=-1)

  pred_ks_log = pred_ks.copy()
  pred_ks_log[pred_ks_log == 0] = 10e-4
  pred_logk = np.log(pred_ks_log)
  pred_trajs_log = np.concatenate((pred_logk, pred_alphas), axis=-1)

  states=1
  ensembles_alphas = []
  ensembles_ks = []
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
    for id_part, ran in enumerate(fovs_rang_trajs[img]):
        if ch_mode == 0:
          dataK = pred_ks[img, id_part][ran[0]:ran[1]].flatten()
          Q2 = np.percentile(dataK, 50)
          if Q2 > 3:
            cp_part = change_pointslog(pred_trajs_log[img, id_part], ran[0], ran[1])
          else:
            cp_part = change_pointsnolog(pred_trajs[img, id_part], ran[0], ran[1])
        else:
          cp_part = change_points_model(pred_states[img, id_part], ran[0], ran[1])

        if cp_part[-1] == ran[1]-ran[0]:
          cp_part= cp_part[:-1]

        if len(cp_part) == 0:
          alpha = np.median(pred_alphas[img,id_part, ran[0]:ran[1]])
          ks = np.median(pred_ks[img,id_part, ran[0]:ran[1]])
          state = Counter(pred_states[img,id_part, ran[0]:ran[1]]).most_common(1)[0][0]
          state =  state-1

          if alpha > 1.9:
            state=3
          if state==1:
            alpha=0
            ks=0

          ensembles_alphas += [alpha]*int(ran[1]-ran[0]+1)
          ensembles_ks += [ks]*int(ran[1]-ran[0]+1)
          res = [int(id_part+add_ind), ks, alpha, int(state), int(ran[1]-ran[0]+1)]
        else:
          states = 2
          alpha = np.median(pred_alphas[img,id_part, ran[0]:cp_part[0]+ran[0]])
          ks = np.median(pred_ks[img,id_part, ran[0]:cp_part[0]+ran[0]])
          state = Counter(pred_states[img,id_part, ran[0]:ran[1]]).most_common(1)[0][0]
          state =  state-1

          if alpha > 1.9:
            state=3
          if state==1:
            alpha=0
            ks=0

          ensembles_alphas += [alpha]*int(cp_part[0]+1)
          ensembles_ks += [ks]*int(cp_part[0]+1)
          preb_chcount = int(cp_part[0]+1)

          res = [int(id_part+add_ind), ks, alpha, state, int(cp_part[0]+1)]
          cp_part = cp_part+[int(ran[1]-ran[0])]
          for i in range(len(cp_part)-1):
            state = Counter(pred_states[img,id_part, ran[0]:ran[1]]).most_common(1)[0][0]
            state =  state-1


            alpha = np.median(pred_alphas[img,id_part,int(cp_part[i]+ran[0]):int(cp_part[i+1]+ran[0])])
            ks = np.median(pred_ks[img,id_part,int(cp_part[i]+ran[0]):int(cp_part[i+1]+ran[0])])
            if alpha > 1.9:
              state=3

            if state==1:
              alpha=0
              ks=0

            ensembles_alphas += [alpha]*(int(cp_part[i+1]+1)-preb_chcount)
            ensembles_ks += [ks]*(int(cp_part[i+1]+1)-preb_chcount)
            preb_chcount = int(cp_part[i+1]+1)

            res.extend([ks, alpha, state, int(cp_part[i+1]+1)])

  with open(dir_pred + "/ensemble_labels.txt", 'w') as f:
    if states == 1:
      f.write(f'model: single_state; num_state: {int(states)} \n')
      data = []
      data.append(np.array([np.mean(np.array(ensembles_alphas)), np.std(np.array(ensembles_alphas)),
                          np.mean(np.array(ensembles_ks)), np.std(np.array(ensembles_ks)), len(ensembles_alphas)]))
    else:
      state_type = "multi_state"

      if ch_mode==1:
        if states_stats[0][1] == 1:
          state_type = "immobile_traps"
        if states_stats[0][1] == 2:
          state_type = "confinement"

      f.write(f'model: multi_state; num_state: {int(states)} \n')
      cluster_states = k_means_clusters(ensembles_alphas, ensembles_ks)
      data = []
      data.append(np.array([np.mean(np.array(cluster_states[0][0])), np.std(np.array(cluster_states[0][0])),
                            np.mean(np.array(cluster_states[0][1])), np.std(np.array(cluster_states[0][1])), len(cluster_states[0][0])]))
      data.append(np.array([np.mean(np.array(cluster_states[1][0])), np.std(np.array(cluster_states[1][0])),
                            np.mean(np.array(cluster_states[1][1])), np.std(np.array(cluster_states[1][1])), len(cluster_states[1][0])]))


    data = np.transpose(np.array(data))
    data[-1,:] /= data[-1,:].sum()
    # Save the data in the corresponding ensemble file
    np.savetxt(f, data, delimiter = ';')


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



