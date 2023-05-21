
## Imports ##
import os
import cv2
import time
import math
import torch
import pickle
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms


### Helper functions ###

def generate_boxes_landmarks(img, mtcnn, device):
    all_boxes, all_probs, all_landmarks = mtcnn.detect(torch.Tensor(img).to(device), landmarks=True)
    if all_boxes is None: return [], [], []
    all_boxes = [[int(x) for x in box] for box in all_boxes] 
    all_landmarks = [[[int(x), int(y)] for x, y in point] for point in all_landmarks] 

    boxes, probs, landmarks, centres = [], [], [], []
    threshold = 0.9
    for box, prob, landmark in zip(all_boxes, all_probs, all_landmarks):
            if prob >= threshold:
                boxes.append(box)
                probs.append(prob)
                landmarks.append(landmark)
    # print(f"[{len(boxes)}/{len(all_boxes)}] faces used")
    return boxes, landmarks, probs

def calculate_rotate_angle(left_eye, right_eye):

    if left_eye[1] > right_eye[1]: # right eye higher than left eye
        # print("rotating clockwise")
        direction = -1
        third_point = (right_eye[0], left_eye[1])
    else:
        # print("rotating counter-clockwise")
        direction = 1
        third_point = (left_eye[0], right_eye[1])

    def euclidean_distance(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    a = euclidean_distance(left_eye, third_point)
    b = euclidean_distance(right_eye, left_eye)
    c = euclidean_distance(right_eye, third_point)
    angle = np.degrees(np.arccos((b*b + c*c - a*a)/(2*b*c)))

    if direction == -1:
        angle = 90 - angle

    rotate_angle = direction * angle
    return rotate_angle

def find_new_bbox_cords(mtcnn, rotated_img, face_centre, device):
    new_boxes, _, _ = generate_boxes_landmarks(rotated_img, mtcnn, device)
    boxes_distances = []
    for new_box in new_boxes:
        centre = [(new_box[0] + new_box[2])//2, (new_box[1] + new_box[3])//2]
        difference = abs(np.array(centre) - np.array(face_centre)).mean()
        boxes_distances.append([difference, new_box])
    if len(boxes_distances) == 0: 
        return []
    boxes_distances.sort(key=lambda x: x[0])
    wanted_box = boxes_distances[0][1]
    return wanted_box

def poisson_blend(paste_image, source_img, box):
    """ Poisson blending using seamlessClone
    blends paste_image into source_image
    """
    src_mask = np.zeros(paste_image.shape, paste_image.dtype)
    height,width = paste_image.shape[:2]

    rectangle = np.array([
        [0, 0], 
        [0, height],
        [width, height],
        [width, 0]], np.int32)
    cv2.fillPoly(src_mask, [rectangle], (255, 255, 255))

    box_centre = [(box[0] + box[2])//2, (box[1] + box[3])//2]
    blended = cv2.seamlessClone(paste_image, source_img, src_mask, box_centre, cv2.NORMAL_CLONE)
    return blended

##### Inpainting #####

def train(G, D, fixed_noise, cropped_real_face_tensor, mask, lr = 0.0003, iterations = 1500, lam = 0.1, eval_interval = 200):

    #lam is perceptual_loss factor
    progress = []
    fixed_noise = fixed_noise.clone().requires_grad_(True)

    # criterion = nn.BCELoss()
    optimizer = optim.Adam([fixed_noise], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    t_start = time.time()
    for i in range(iterations):
        fake_face = G(fixed_noise, None)
        contextual_loss = nn.functional.l1_loss(mask*fake_face, mask*cropped_real_face_tensor)
        perceptual_loss = D(fake_face, None)[0][0] # is unbounded. 0 is a awful prediction, more negative means more confident its a face

        complete_loss = contextual_loss + lam*perceptual_loss

        optimizer.zero_grad()
        complete_loss.backward()
        optimizer.step()
        scheduler.step()

        if i % eval_interval == 0 or i == iterations-1:
            print(f"Losses, {i} iteration:: Complete:{complete_loss:.4f}, contextual:{contextual_loss:.4f}, perceptual:{lam*perceptual_loss:.4f} (after x0.1), time: {time.time()-t_start:.2f}s")
            progress.append((cropped_real_face_tensor*mask+fake_face*(1-mask)).cpu())
    return progress

def tensor_to_np(img):
    """Torch tensor -> normalized np image"""
    torch_grid = torchvision.utils.make_grid(img.cpu(), normalize = True, padding = 0)
    return np.ascontiguousarray((torch_grid.permute(1,2,0).numpy()*255), dtype=np.uint8)

def inpaint(G, D, mtcnn, device, cropped_face, fixed_noise, lr = 0.0003, iterations = 1500, lam = 0.1, eval_interval = 200, display_intermediate = False,  border_factor = 0.15):

    ## Generate image             
    generated_img_tensor = G(fixed_noise, None)  # NCHW, float32, dynamic range [-1, +1], None is class labels
    generated_img = tensor_to_np(generated_img_tensor)

    #finds boxes, landmarks using generated image
    generated_boxes, generated_landmarks, generated_probs = generate_boxes_landmarks(generated_img, mtcnn, device)
    #Add loop to create different generated face if this is the case
    assert len(generated_boxes) >=1, "No faces detected in generated image"
    assert len(generated_boxes) == 1, "Two faces detected in generated image"
    box_generated = generated_boxes[0]
    x1, y1, x2, y2 = box_generated
    width = x2 - x1
    height = y2 - y1
    print(f"width: {width}, height: {height}")

    border_factor = 0.15
    tensor_transform = transforms.ToTensor()

    border_width = int(width*border_factor)
    border_height = int(height*border_factor)
    mask = torch.zeros((1, 3, 1024, 1024)).to(device)
    mask[:, :, y1:y2, x1:x2] = 1
    mask[:, :, y1+border_height:y2-border_height, x1+border_width:x2-border_width] = 0

    #convert real face to torch tensor, place aligned with generated face in 1024x1024 square
    resized_face = cv2.resize(cropped_face.copy(), [width, height])
    cropped_real_face_tensor = torch.zeros((1, 3, 1024, 1024)).to(device)
    cropped_real_face_tensor[:, :, y1:y2, x1:x2] = tensor_transform(resized_face).unsqueeze(dim=0).to(device)

    if display_intermediate:
        images_to_concat = []
        for image in [cropped_real_face_tensor, generated_img_tensor, mask,mask*cropped_real_face_tensor, mask*generated_img_tensor, cropped_real_face_tensor*mask+generated_img_tensor*(1-mask)]:
            images_to_concat.append(tensor_to_np(image))
        display_img = np.concatenate(images_to_concat,axis=1)
        display(ImageOps.contain(Image.fromarray(display_img), (1500, 1500)))

    ## Train ##
    progress = train(G, D, fixed_noise, cropped_real_face_tensor, mask, lr = lr, iterations = iterations, lam = lam, eval_interval = eval_interval)

    if display_intermediate:
        images_to_concat = []
        for image in [cropped_real_face_tensor, mask, progress[-1]]:
            images_to_concat.append(tensor_to_np(image))
        display_img = np.concatenate(images_to_concat,axis=1)
        display(ImageOps.contain(Image.fromarray(display_img), (1500, 1500)))

    
    inpainted_faces = []
    for inpainted_face in progress:
        inpainted_face = tensor_to_np(inpainted_face[:, :, y1:y2, x1:x2])
        inpainted_face = cv2.resize(inpainted_face, (cropped_face.shape[1], cropped_face.shape[0]))

        border_width = int(inpainted_face.shape[1]*border_factor)
        border_height = int(inpainted_face.shape[0]*border_factor)
        inpainted_face = inpainted_face[border_height:-border_height, border_width:-border_width]
        inpainted_faces.append(inpainted_face)
    return inpainted_faces
