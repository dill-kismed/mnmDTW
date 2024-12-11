import pandas as pd
import numpy as np
from scipy.stats import zscore
from dtaidistance import dtw
from dtaidistance import dtw_ndim
from dtaidistance import dtw_visualisation as dtwvis
import matplotlib.pyplot as plt
import os

# set data path
current_path = os.path.dirname(os.path.abspath(__file__))
path_csv = current_path + r"DATA\mediapipe_output.csv"

# set information about the joints present in the data. See MediaPipe documentation for details about the joints.
joints = np.arange(0,33)
df = pd.read_csv(path_csv)
joint_names_tmp = df.columns[joints*4+2]
joint_names = []
for jn in joint_names_tmp:
    joint_names.append(jn[:-2])

# data from several videos can be present in one CSV-File. Each video is processed separately.
# There are three videos in the available data: 0 shows correct repetitions, 1 shows feet that are too wide, 2 shows an execution where the subject does not go down far enough.
video_name_list = df['filename'].unique()
split_data_list = []
normalized_data_list = []
for idx, video_name in enumerate(video_name_list):
    current_df = df.loc[df['filename'] == video_name]
    current_df = current_df.drop('filename',axis=1)
    delta_t = current_df['timestamp [ms]'].iloc[1]-current_df['timestamp [ms]'].iloc[0]
    current_df['frame_number'] = (current_df['timestamp [ms]']/delta_t).astype('int')
    # hard coded time stamps of the different repetitions
    if idx == 0:
        splits = [0, 180, 420, 660, 870, 1050, 1290, 1500, 1680, 1890, -1] # correct
    if idx == 1:
        splits = [0, 180, 360, 520, 690, -1] # feet too wide
    if idx == 2:
        splits = [0, 210, 360, 510, -1] # not far enough down
    split_data = []
    normalized_split_data = []
    for i in range(len(splits[:-1])):
        tmp = np.hstack((np.array([current_df.iloc[splits[i]:splits[i+1]].iloc[:,joints*4+1]]).squeeze(),np.array([current_df.iloc[splits[i]:splits[i+1]].iloc[:,joints*4+2]]).squeeze())) # only use x and y coordinates, ignore z coordinate and visibility
        split_data.append(tmp)
        normalized_df = current_df.apply(zscore) # normalization is needed for DTW to work
        normalized_split_data.append(np.array([normalized_df.iloc[splits[i]:splits[i+1]].iloc[:,joints*4+1],normalized_df.iloc[splits[i]:splits[i+1]].iloc[:,joints*4+2]])) # only use x and y coordinates, ignore z coordinate and visibility
    split_data_list.append(split_data)
    normalized_data_list.append(normalized_split_data)
# lists are structured in the following manner:
# [exercise type][example nr.][frame,joint (first 33 x-coords, then 33 y-coords)]
# sets of joints so they are logical groups:
head = list(range(0,11))
torso = [11,12,23,24]
left_arm = [13,15,17,19,21]
right_arm = [14,16,18,20,22]
left_leg = [25,27,29,31]
right_leg = [24,26,28,30]
limbs = [head,torso,left_arm,right_arm,left_leg,right_leg]

mean_for_correct_list = []
for t in range(1,6):
    s1 = split_data_list[0][0][:,:]
    s2 = split_data_list[0][t][:,:]
    md, mpaths = dtw_ndim.warping_paths(s1,s2)
    best_mpath = dtw.best_path(mpaths)
    idtw_list = []
    for i in range(2):
        for limb in limbs:
            si1 = []
            si2 = []
            for j in range(len(best_mpath)):
                si1.append(s1[best_mpath[j][0],33*i+np.array(limb)])
                si2.append(s2[best_mpath[j][1],33*i+np.array(limb)])
            # si1 = s1[:,33*i+np.array(limb)]
            # si2 = s2[:,33*i+np.array(limb)]
            idtw_list.append(dtw_ndim.distance(si1,si2))
    mean_for_correct_list.append(idtw_list)
mean_for_correct = np.mean(np.array(mean_for_correct_list),axis=0)
for e in range(3):
    for t in range(4):
        if e == 0:
            t += 6
        s1 = split_data_list[0][0][:,:]
        s2 = split_data_list[e][t][:,:]
        md, mpaths = dtw_ndim.warping_paths(s1,s2)
        best_mpath = dtw.best_path(mpaths)
        idtw_list = []
        for i in range(2):
            for limb in limbs:
                si1 = []
                si2 = []
                for j in range(len(best_mpath)):
                    si1.append(s1[best_mpath[j][0],33*i+np.array(limb)])
                    si2.append(s2[best_mpath[j][1],33*i+np.array(limb)])
                # si1 = s1[:,33*i+np.array(limb)]
                # si2 = s2[:,33*i+np.array(limb)]
                idtw_list.append(dtw_ndim.distance(si1,si2))
        normalized_idtw = idtw_list/mean_for_correct
        f,ax = plt.subplots()
        f.set_figheight(6)
        f.set_figwidth(8)
        f.tight_layout
        width = 0.2
        ax.bar(np.arange(0,6),normalized_idtw[0:6],width=width)
        ax.bar(np.arange(0,6)+width,normalized_idtw[6:],width=width)
        ax.set_xticks(np.arange(0,6)+0.5*width,labels=['head','torso','left_arm','right_arm','left_leg','right_leg'],fontsize=14)
        ax.set_ylim(0,20)
        plt.yticks(fontsize=14)
        ax.legend(('x','y'),loc='upper left',fontsize=18)
        f.savefig(current_path + r"FIGURES" + r"\test_exercise" + str(e) + "_sample" + str(t) + ".png")


# s1 = split_data_list[0][0][:,23]
# s2 = split_data_list[2][2][:,23]
# d, paths = dtw.warping_paths(s1,s2)
# best_path = dtw.best_path(paths)
# dtwvis.plot_warpingpaths(s1,s2,paths,best_path)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.show()
