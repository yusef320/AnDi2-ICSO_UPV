from sklearn.cluster import KMeans
import numpy as np
import ruptures as rpt
from collections import Counter
from scipy.optimize import linear_sum_assignment
from andi_datasets.utils_videos import import_tiff_video
import trackpy as tp
import pandas as pd
from scipy.optimize import linear_sum_assignment
import keras


def expand_borders(v, fm=False):
    expanded_images = []
    for frame in v:
      if fm:
        img_median = 255
      else:
        img_median = np.median(frame)

      img_borders = np.vstack((frame[-4:, :], frame, frame[:4, :]))
      img_borders = np.hstack((img_borders[:, -4:], img_borders, img_borders[:, :4]))
      img_borders[:4, :] = img_median
      img_borders[-4:, :] = img_median
      img_borders[:, :4] = img_median
      img_borders[:, -4:] = img_median
      expanded_images.append(img_borders)

    return np.array(expanded_images)

def video2traj(raw_video):
    raw_video = expand_borders(raw_video)
    f = tp.batch(raw_video, diameter=3, invert=True, minmass=13 ,separation=2.6, max_iterations=10)
    try:
      traj = tp.link(f, 25, memory=10, neighbor_strategy="BTree", link_strategy="auto")
    except:
      traj = tp.link(f, 4, memory=10, neighbor_strategy="BTree", link_strategy="auto")
    return traj[["particle","frame","x","y"]].sort_values(by=["particle","frame"], ignore_index=True)

def loc_particles(firstframe, df_video):
    indxparticles = {}
    df_firstframe = df_video[df_video["frame"]==0]
    firstframe = expand_borders([firstframe])[0]
    items = np.unique(firstframe)[:-1]
    positions = []
    for itm in items:
        pos = np.where(firstframe==itm)
        positions.append((np.mean(pos[0]), np.mean(pos[1])))

    positions_array = np.array(positions)
    detected_array = df_firstframe[['y', "x"]].values
    distances = np.sqrt(((detected_array[:, np.newaxis, :] - positions_array) ** 2).sum(axis=2))
    row_ind, col_ind = linear_sum_assignment(distances)
    indxparticles = {}


    for detected, asigned in zip(row_ind, col_ind):
      if len(np.where(firstframe==items[asigned])[0]) >= 3:
        indxparticles[int(detected)] = int(items[asigned])
    return indxparticles

def load_videos(dir, n_fov):
    fovs = []
    vipfovs = []
    print("Loading videos ...")
    for fov in range(n_fov):
      print(f"Video {fov}")
      video = import_tiff_video(dir+f"videos_fov_{fov}.tiff")
      vidtrajs = video2traj(video[1:])
      locpart = loc_particles(video[0], vidtrajs)
      fovs.append(vidtrajs)
      vipfovs.append(locpart)
    print("Videos loaded")
    return fovs, vipfovs

def read_fovs(nfov, dir_data):
  fovs_trajs = []
  fovs_trajs_2 = []
  fovs_id = []
  fovs_rang_trajs = []
  fovs_ind_trajs = []
  fovs_video, vipfovs = load_videos(dir_data, nfov)

  for fov, df in enumerate(fovs_video):
    fovs_id.append(fov)
    input = np.zeros((64,208, 2))
    rang_trajs = []
    real_ind = []
    grouped = df.groupby(['particle'])
    indvips = 0
    for _,group in grouped:
      part = int(_[0])
      org = int(group["frame"].iloc[0])
      dest = org+len(group["x"])
      if part in vipfovs[fov].keys():
        rang_trajs.append((org, dest))
        real_ind.append(vipfovs[fov][part])
        input[indvips, org:dest, 0] = group["x"]
        input[indvips, org:dest, 1] = group["y"]
        indvips += 1

    for i in range(1,208):
        input[:,-i] = input[:,-i]-input[:,-i-1]

    if len(real_ind) != 10:
      print(f"len real {len(real_ind)}")
    fovs_ind_trajs.append(real_ind)
    fovs_rang_trajs.append(rang_trajs)
    input_copy = input.copy()
    input[:,0] = np.zeros((64, 2))
    fovs_trajs.append(input)
    fovs_trajs_2.append(input_copy)

  return fovs_trajs,fovs_trajs_2,fovs_rang_trajs, fovs_id, vipfovs, fovs_ind_trajs

def normalizar_pred(signal):
  for i in range(1, len(signal)-2):
    if signal[i-1] != signal[i] and (signal[i+1] != signal[i] or signal[i+2] != signal[i]) :
      signal[i] = signal[i-1]
  return signal

def change_points_model(pred_states, org, dest):
  try:
    signal = normalizar_pred(pred_states[org:dest])
    model = "l2"  # Model used for segmentation
    algo = rpt.Window(width = 10, model=model, jump = 1, min_size = 3).fit(signal)
    cp = algo.predict(pen=1)
    return cp
  except:
    return []


def change_pointsnolog(pred_coef, org, dest):
  try:
    signal = pred_coef[org:dest]
    model = "l2"  # Model used for segmentation
    algo = rpt.Window(width = 18, model=model, jump = 1, min_size = 3).fit(signal)
    cp = algo.predict(pen = 5)
    return cp
  except:
    return []

def k_means_clusters(lista1, lista2):
    # Combinar las dos listas en un array 2D
    data = list(zip(lista1, lista2))

    # Aplicar K-Means con K=2
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(data)

    # Obtener las etiquetas de los clusters
    labels = kmeans.labels_

    # Agrupar los datos por cluster
    clusters = [[[],[]],[[],[]]]
    for i, label in enumerate(labels):
        clusters[label][0].append(lista1[i])
        clusters[label][1].append(lista2[i])
    return clusters

def missing_parts(archivo, fov, exp, dir_data):
    files_missing = []
    try:
        with open(archivo, 'r', encoding='utf-8') as file:
            lineas = file.readlines()
            if len(lineas) != 10:
                files_missing.append((archivo, dir_data+f""))
    except:
       pass

def pred_trajs_fov_video(nfov, dir_pred, dir_data, unet_alpha, unet_ks, unet_states):
  fovs_trajs,fovs_trajs2, fovs_rang_trajs, fovs_id, vips, fovs_ind_trajs = read_fovs(nfov, dir_data)

  print("Predicting videos")
  pred_alphas = unet_alpha.predict(np.array(fovs_trajs), verbose=0)
  pred_ks = unet_ks.predict(np.array(fovs_trajs), verbose=0)
  pred_trajs = np.concatenate((pred_ks, pred_alphas), axis=-1)
  pred_states = np.argmax(unet_states.predict(np.array(fovs_trajs2), verbose=0), axis=-1)

  states=1
  ensembles_alphas = []
  ensembles_ks = []
  preb = -1

  states_stats = np.unique(pred_states, return_counts=True)
  ch_mode = 0
  if len(states_stats[1]) == 3:
    if states_stats[1][1] > 200:
      ch_mode = 1

  for fov in range(nfov):
    submission_file = dir_pred + f'/fov_{fov}.txt'
    with open(submission_file, "w") as f:
      print(fovs_rang_trajs)
      for id_part, ran in enumerate(fovs_rang_trajs[fov]):
        img = fov
        if ch_mode==0:
            cp_part = change_pointsnolog(pred_trajs[img, id_part], ran[0], ran[1])
        else:
          cp_part = change_points_model(pred_states[img, id_part], ran[0], ran[1])

        if len(cp_part) > 0:
          if cp_part[-1] == ran[1]-ran[0]:
            cp_part= cp_part[:-1]

        if len(cp_part) == 0:
          alpha = np.median(pred_alphas[img,id_part, ran[0]:ran[1]])
          ks = np.median(pred_ks[img,id_part, ran[0]:ran[1]])
          state = Counter(pred_states[img,id_part, ran[0]:ran[1]]).most_common(1)[0][0]
          state =  state-1
          if state == 1 :
              alpha = 0
              ks = 0
          if state < 0:
            print(state)
          ensembles_alphas += [alpha]*int(ran[1]-ran[0]+1)
          ensembles_ks += [ks]*int(ran[1]-ran[0]+1)
          res = [fovs_ind_trajs[fov][id_part], ks, alpha, int(state), int(ran[1]-ran[0]+1)]
          formatted_numbers = ','.join(map(str, res))
          f.write(formatted_numbers + '\n')
        else:
          states = 2
          alpha = np.median(pred_alphas[img,id_part, ran[0]:cp_part[0]+ran[0]])
          ks = np.median(pred_ks[img,id_part, ran[0]:cp_part[0]+ran[0]])
          state = Counter(pred_states[img,id_part, ran[0]:cp_part[0]+ran[0]]).most_common(1)[0][0]
          state =  state-1
          if state == 1:
            alpha = 0
            ks = 0


          ensembles_alphas += [alpha]*int(cp_part[0]+1)
          ensembles_ks += [ks]*int(cp_part[0]+1)
          preb_chcount = int(cp_part[0]+1)

          res = [fovs_ind_trajs[fov][id_part], ks, alpha, state, int(cp_part[0]+1)]
          cp_part = cp_part+[int(ran[1]-ran[0])]
          for i in range(len(cp_part)-1):
            state = Counter(pred_states[img,id_part,int(cp_part[i]+ran[0]):int(cp_part[i+1]+ran[0])]).most_common(1)[0][0]
            state =  state-1
            alpha = np.median(pred_alphas[img,id_part,int(cp_part[i]+ran[0]):int(cp_part[i+1]+ran[0])])
            ks = np.median(pred_ks[img,id_part,int(cp_part[i]+ran[0]):int(cp_part[i+1]+ran[0])])
            if state == 1 and alpha > 1.7:
              alpha = 0
              ks = 0

            ensembles_alphas += [alpha]*(int(cp_part[i+1]+1)-preb_chcount)
            ensembles_ks += [ks]*(int(cp_part[i+1]+1)-preb_chcount)
            preb_chcount = int(cp_part[i+1]+1)

            res.extend([ks, alpha, state, int(cp_part[i+1]+1)])
          formatted_numbers = ','.join(map(str, res))
          f.write(formatted_numbers + '\n')

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


def comprobar_lineas(archivo):
    with open(archivo, 'r', encoding='utf-8') as file:
        lineas = file.readlines()
        if len(lineas) != 10:
            return True
        return False

def solve_missing_particles(pred_dir, data_dir, fovs):
    for fov in range(fovs):
      misspart = comprobar_lineas(pred_dir+f'/fov_{fov}.txt')
      if misspart:
        video = import_tiff_video(data_dir+ f"/videos_fov_{fov}.tiff")
        particles = np.unique(video[0])[:-1]
        with open(pred_dir+f'/fov_{fov}.txt', 'r', encoding='utf-8') as file:
          lineas = file.readlines()
          print(lineas)
          par_in = []
          for l in lineas:
            par_in.append(int(l.split(",")[0]))
          lost_ids = []
          for i in particles:
            if i not in par_in:
              lost_ids.append(i)

        with open(pred_dir+f'/fov_{fov}.txt', 'a', encoding='utf-8') as f:
          for part in lost_ids:
            print(part)
            res = [part, 0, 0, 2, 5]
            formatted_numbers = ','.join(map(str, res))
            f.write(formatted_numbers + '\n')


unet_alpha = keras.models.load_model("/content/drive/MyDrive/att_unet/alphas-3-2-1024-v5-epoch-2.keras", compile=False)
unet_ks = keras.models.load_model(f"/content/drive/MyDrive/att_unet/ks-3-2-1024-newdata-epoch-3.keras", compile=False)
unet_states = keras.models.load_model(f"/content/drive/MyDrive/att_unet/states-3-6-128-v1-epoch-10.keras", compile=False)

for exp in range(12):
  print("Pred. file ", exp)
  dir_data = rf"/content/drive/MyDrive/public_data_challenge_v0/track_1/exp_{exp}/"
  dir_pred = rf"/content/drive/MyDrive/Model_015/track_1/exp_{exp}"
  pred_trajs_fov_video(30, dir_pred, dir_data, unet_alpha, unet_ks, unet_states)
  solve_missing_particles(dir_pred, dir_data, 30)

