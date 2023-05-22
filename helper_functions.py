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

def generate_inpainting_inputs(G, D, mtcnn, device, cropped_face, fixed_noise, border_factor = 0.15):
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

    generated_face = generated_img[y1:y2, x1:x2]
    resized_face = cv2.resize(generated_face, (cropped_face.shape[1], cropped_face.shape[0]))
    border_width = int(resized_face.shape[1]*border_factor)
    border_height = int(resized_face.shape[0]*border_factor)
    resized_face = resized_face[border_height:-border_height, border_width:-border_width]

    return cropped_real_face_tensor, mask, box_generated, resized_face
    
def inpaint(G, D, mtcnn, device, cropped_face, fixed_noise, cropped_real_face_tensor, mask, box_generated, lr = 0.0003, iterations = 1500, lam = 0.1, eval_interval = 200,  border_factor = 0.15):

    ## Train ##
    progress = train(G, D, fixed_noise, cropped_real_face_tensor, mask, lr = lr, iterations = iterations, lam = lam, eval_interval = eval_interval)

    inpainted_faces = []
    x1, y1, x2, y2 = box_generated
    for inpainted_face in progress:
        inpainted_face = tensor_to_np(inpainted_face[:, :, y1:y2, x1:x2])
        inpainted_face = cv2.resize(inpainted_face, (cropped_face.shape[1], cropped_face.shape[0]))

        border_width = int(inpainted_face.shape[1]*border_factor)
        border_height = int(inpainted_face.shape[0]*border_factor)
        inpainted_face = inpainted_face[border_height:-border_height, border_width:-border_width]
        inpainted_faces.append(inpainted_face)
    return inpainted_faces

def visualize_progress(original_img_padded, box, left_eye, right_eye, rotate_angle, rotated_img, rotated_box, inpainted_faces):
    #1. Draw original face box on
    annotated_faces_img = original_img_padded.copy()
    cv2.rectangle(annotated_faces_img, box[:2], box[2:], (255, 0, 0), 2)
    save_image(annotated_faces_img, pad_width)

    #2. Drawing eyes
    left_eye, right_eye = landmark[0], landmark[1]
    cv2.circle(annotated_faces_img, left_eye, 3, (0,255,255), -1)
    save_image(annotated_faces_img, pad_width)
    cv2.circle(annotated_faces_img, right_eye, 3, (0,255,255), -1)
    save_image(annotated_faces_img, pad_width)

    #3. Creating and drawing 3rd point to create triangle
    if left_eye[1] > right_eye[1]: # right eye higher than left eye
        third_point = (right_eye[0], left_eye[1]) # rotating clockwise
    else:
        third_point = (left_eye[0], right_eye[1]) # rotating counter-clockwise
    cv2.circle(annotated_faces_img, third_point, 3, (0,255,255), -1)
    save_image(annotated_faces_img, pad_width)

    #4. Drawing triangle
    for pair in [[left_eye, right_eye], [left_eye, third_point], [right_eye, third_point]]:
        cv2.line(annotated_faces_img, pair[0], pair[1], (0,255,0), 1)
    save_image(annotated_faces_img, pad_width)

    #5. Rotating image, writing angle on it
    rotated_img_annotated = np.array(Image.fromarray(annotated_faces_img).rotate(rotate_angle, center=face_centre, expand=False))
    cv2.putText(rotated_img_annotated, f"Rotating by {rotate_angle:.2f}° anti-clockwise", (pad_width+10,pad_width+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    save_image(rotated_img_annotated, pad_width)

    #6. Draw on new face on fulled labelled image
    cv2.rectangle(rotated_img_annotated, rotated_box[:2], rotated_box[2:], (0, 255, 0), 2)
    save_image(rotated_img_annotated, pad_width)

    #7. Draw face on tidy image
    rotated_img_annotated = rotated_img.copy()
    cv2.rectangle(rotated_img_annotated, rotated_box[:2], rotated_box[2:], (0, 255, 0), 2)
    save_image(rotated_img_annotated, pad_width)

    #8. Perform Facial obfuscation 
    x1, y1, x2, y2 = rotated_box
    width = x2 - x1
    height = y2 - y1
    width_border = int(width * border_factor)
    height_border = int(height * border_factor)
    rotated_img_obscured = rotated_img_annotated.copy()
    cv2.rectangle(rotated_img_obscured, [x1+width_border, y1+height_border], [x2-width_border-1, y2-height_border-1], (255, 255, 255), -1)
    save_image(rotated_img_obscured, pad_width)

    #9. obfuscation image rotated to original orientation
    unrotated_img_annotated = np.array(Image.fromarray(rotated_img_obscured).rotate(-rotate_angle, center=face_centre, expand=False))
    save_image(unrotated_img_annotated, pad_width)


    #10. Place raw generated face onto image
    rotated_img_annotated_raw = rotated_img_annotated.copy()
    start_x, start_y = rotated_box[1]+height_border, rotated_box[0]+width_border
    rotated_img_annotated_raw[start_x:start_x+inpainted_faces[0].shape[0], start_y:start_y+inpainted_faces[0].shape[1]] = inpainted_faces[0] # naive replacement
    unrotated_img_annotated_raw = np.array(Image.fromarray(rotated_img_annotated_raw).rotate(-rotate_angle, center=face_centre, expand=False))
    save_image(unrotated_img_annotated_raw, pad_width)

    #11. Poisson blend and show progress
    for inpainted_face in inpainted_faces:
        rotated_img_blended = rotated_img_annotated.copy()
        rotated_img_blended = poisson_blend(inpainted_face, rotated_img_blended, rotated_box)
        unrotated_img_blended = np.array(Image.fromarray(rotated_img_blended).rotate(-rotate_angle, center=face_centre, expand=False))
        save_image(unrotated_img_blended, pad_width)
