import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick

simulation_phase = []
rendering_phase = []

collision_detection = []
fem_constraint = []
collision_constraint = []
xpbd_update = []
gs_embedding = []

shadow_map = []
left_eye = []
right_eye = []

fox = [0, 454, 0, 456, 47, 0, 2992, 3323]
bear = [0, 1902, 0, 1316, 121, 3360, 9668, 9636]
horse = [0, 1104, 0, 1124, 145, 0, 6805, 7674]
ring_toss = [1895, 3704, 2561, 6759, 130, 14687, 4294, 4246]
table_game = [6908, 5529, 2066, 3039, 234, 6759, 21426, 20413]
toy_collection = [9832, 20062, 4963, 4655, 223, 5583, 23794, 22810]
box_moving = [0, 142, 0, 176, 103, 0, 6213, 6553]
animal_crossing = [9470, 13088, 1995, 4498, 119, 4438, 15319, 13245]
just_dance = [0, 1362, 0, 911, 54, 6328, 12933, 12835]

labels = ['Fox', 'Bear', 'Horse', 'Ring Toss', 'Table Brick Game', 'Toy Collection', 'Box Moving', 'Animal Crossing', 'Just Dance']

time_break_list = [fox, bear, horse, ring_toss, table_game, toy_collection, box_moving, animal_crossing, just_dance]

labels.reverse()
time_break_list.reverse()

barWidth = 0.3
br1 = np.arange(len(time_break_list)) + barWidth
br2 = [x + barWidth for x in br1]

for time_break in time_break_list:
    assert(len(time_break) == 8)
    time = np.array(time_break)
    time = time / np.sum(time)
    
    collision_detection.append(time[0])
    fem_constraint.append(time[1])
    collision_constraint.append(time[2])
    xpbd_update.append(time[3])
    gs_embedding.append(time[4])
    simulation_phase.append(time[0] + time[1] + time[2] + time[3] + time[4])

    shadow_map.append(time[5])
    left_eye.append(time[6])
    right_eye.append(time[7])
    rendering_phase.append(time[5] + time[6] + time[7])

# plt.style.use('ggplot')
custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False, "font.size": 15}
matplotlib.rc('font', weight='bold')
sns.set_theme(style='ticks', rc=custom_params, font_scale=1.2)
plt.figure(figsize=(20, 5))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

plt.barh(br1, collision_detection, height=barWidth, color=['olive'], label='Collision Detection')
left_sum = np.array(collision_detection)

plt.barh(br1, fem_constraint, left=left_sum, height=barWidth, color=['blue'], label='FEM Constraint')
left_sum += fem_constraint

plt.barh(br1, collision_constraint, left=left_sum, height=barWidth, color=['red'], label='Collision Constraint')
left_sum += collision_constraint

plt.barh(br1, xpbd_update, left=left_sum, height=barWidth, color=['orange'], label='XPBD Update')
left_sum += xpbd_update

plt.barh(br1, gs_embedding, left=left_sum, height=barWidth, color=['navy'], label='GS Embedding Interpolation')
left_sum += gs_embedding

plt.barh(br1, shadow_map, left=left_sum, height=barWidth, color=['blueviolet'], label="Shadow Map Rendering")
left_sum += shadow_map

plt.barh(br1, left_eye, left=left_sum, height=barWidth, color=['purple'], label="Left Eye Rendering")
left_sum += left_eye

plt.barh(br1, right_eye, left=left_sum,height=barWidth,  color=['orangered'], label='Right Eye Rendering')
left_sum += right_eye

plt.barh(br2, simulation_phase, height=barWidth, fill=False, hatch='//', edgecolor=['blue'], label='Simulation Phase')
left_sum = np.array(simulation_phase)

plt.barh(br2, rendering_phase, left=left_sum, height=barWidth, fill=False, hatch='//', edgecolor=['red'], label='Rendering Phase')
left_sum += rendering_phase

plt.yticks([r + 0.45 for r in range(len(labels))], labels)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5)
plt.savefig("timing_breakdown_new.pdf")