import numpy as np
import cv2
from matplotlib import pyplot as plt
import pysift

beans = cv2.cvtColor(cv2.imread('beans.jpg'), cv2.COLOR_BGR2GRAY)
street = cv2.cvtColor(cv2.imread('street.jpg'), cv2.COLOR_BGR2GRAY)
forest = cv2.cvtColor(cv2.imread('forest.jpg'), cv2.COLOR_BGR2GRAY)
images = [beans, street, forest]
beans1 = cv2.cvtColor(cv2.imread('beans1.jpg'), cv2.COLOR_BGR2GRAY)
street1 = cv2.cvtColor(cv2.imread('street1.jpg'), cv2.COLOR_BGR2GRAY)
forest1 = cv2.cvtColor(cv2.imread('forest1.jpg'), cv2.COLOR_BGR2GRAY)
images_dist = [beans1, street1, forest1]

def blur(image):
  kernel_size = (5,5)
  blur = cv2.blur(image, kernel_size)
  return blur

images_dist1 = [blur(image) for image in images_dist]

def plot_images(images, to_rgb):  
  fig = plt.figure(figsize=(20, 100))
  rows = 1
  columns = len(images)

  for i in range(len(images)):
    fig.add_subplot(rows, columns, i+1)

    if (to_rgb):  
      plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_GRAY2RGB))
    else:
      plt.imshow(images[i])

  plt.show()

plot_images(images, True)

plot_images(images_dist1, True)

#Beans detecting and counting
kp, ds = pysift.computeKeypointsAndDescriptors(images[0])
b_kp, b_ds = pysift.computeKeypointsAndDescriptors(images_dist[0])
#Street detecting and counting
kp1, ds1 = pysift.computeKeypointsAndDescriptors(images[1])
b_kp1, b_ds1 = pysift.computeKeypointsAndDescriptors(images_dist[1])
#Forest detecting and counting
kp2, ds2 = pysift.computeKeypointsAndDescriptors(images[2])
b_kp2, b_ds2 = pysift.computeKeypointsAndDescriptors(images_dist[2])

keys = [(kp, b_kp), (kp1, b_kp1), (kp2, b_kp2)]
des = [(ds, b_ds), (ds1, b_ds1), (ds2, b_ds2)]

def match(des1, des2, ratio=0.85): 
    matches1 = []
    matches2 = []
    distances = {}
    
    for i in range(len(des1)):
        des2_des1 = des2-des1[i]
        d = np.linalg.norm(des2_des1, ord=2, axis=1)
        sorted_args = np.argsort(d).tolist()

        if d[sorted_args[0]]/d[sorted_args[1]] <= ratio:
            matches1.append((i,sorted_args[0]))
            distances[f'{i}-{sorted_args[0]}'] = d[sorted_args[0]]
    
    # for cross-check
    for i in range(len(des2)):
        des1_des2 = des1-des2[i]
        d = np.linalg.norm(des1-des2[i], ord=2, axis=1)
        sorted_args = np.argsort(d).tolist()

        if d[sorted_args[0]]/d[sorted_args[1]] <= ratio:
            matches2.append((sorted_args[0],i))
            distances[f'{sorted_args[0]}-{i}'] = d[sorted_args[0]]
    
    match = list(set(matches1).intersection(set(matches2)))
    return [cv2.DMatch(args_pair[0], args_pair[1], distances[f'{args_pair[0]}-{args_pair[1]}']) for args_pair in match]

def match_and_compare(keys, des):
    match_list = match(des[0], des[1])
    custom_matches = sorted(match_list, key= lambda x: x.distance)

    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    flann_matches = sorted(flann.match(des[0], des[1]), key= lambda x: x.distance)

    return custom_matches, flann_matches 

for i in range(3):
    matches = match_and_compare(keys[i], des[i])
    l1 = int(len(matches[0])*0.2)
    l2 = int(len(matches[1])*0.2)
    custom_match_img = cv2.drawMatches(images[i], keys[i][0], images_dist1[i], keys[i][1], matches[0][:l1], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    flann_match_img = cv2.drawMatches(images[i], keys[i][0], images_dist1[i], keys[i][1], matches[1][:l2], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plot_images([custom_match_img, flann_match_img], False)