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


## functions ##

def generate_boxes_landmarks(image, mtcnn, device):
    """
    Generates bounding boxes, landmarks, and probabilities for detected faces in an image using MTCNN.

    Args:
        image (numpy.ndarray): The input image.
        mtcnn: An instance of MTCNN (face detection model).
        device: The device to run the computation on.

    Returns:
        tuple: A tuple containing lists of bounding boxes, landmarks, and probabilities.
    """

    # Detect faces and landmarks using MTCNN
    all_boxes, all_probs, all_landmarks = mtcnn.detect(torch.Tensor(image).to(device), landmarks=True)

    # Return empty lists if no faces are detected
    if all_boxes is None:
        return [], [], []

    # Convert bounding boxes and landmarks to integer values
    all_boxes = [[int(x) for x in box] for box in all_boxes] 
    all_landmarks = [[[int(x), int(y)] for x, y in point] for point in all_landmarks] 

    # Filter boxes based on probability threshold
    threshold = 0.9
    boxes, landmarks, probs = [], [], []
    for box, prob, landmark in zip(all_boxes, all_probs, all_landmarks):
        if prob >= threshold:
            boxes.append(box)
            landmarks.append(landmark)
            probs.append(prob)

    # Return the filtered lists of boxes, landmarks, and probabilities
    return boxes, landmarks, probs

def calculate_rotate_angle(left_eye, right_eye):
    """
    Calculates the rotation angle based on the positions of the left and right eyes.

    Args:
        left_eye (tuple): The position of the left eye in the form (x, y).
        right_eye (tuple): The position of the right eye in the form (x, y).

    Returns:
        float: The rotation angle in degrees.
    """

    if left_eye[1] > right_eye[1]:  # Right eye higher than left eye
        # print("Rotating clockwise")
        direction = -1
        third_point = (right_eye[0], left_eye[1])
    else:
        # print("Rotating counter-clockwise")
        direction = 1
        third_point = (left_eye[0], right_eye[1])

    def euclidean_distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    a = euclidean_distance(left_eye, third_point)
    b = euclidean_distance(right_eye, left_eye)
    c = euclidean_distance(right_eye, third_point)

    angle = np.degrees(np.arccos((b * b + c * c - a * a) / (2 * b * c)))

    if direction == -1:
        angle = 90 - angle

    rotate_angle = direction * angle
    return rotate_angle


def find_new_bbox_coords(mtcnn, rotated_image, face_centre, device):
    """
    Finds the coordinates of the new bounding box after image rotation.

    Args:
        mtcnn: An instance of MTCNN (face detection model).
        rotated_image (numpy.ndarray): The rotated image.
        face_centre (tuple): The coordinates of the face center in the form (x, y).
        device: The device to run the computation on.

    Returns:
        list or tuple: The coordinates of the new bounding box.
    """

    new_boxes, _, _ = generate_boxes_landmarks(rotated_image, mtcnn, device)
    boxes_distances = []
    for new_box in new_boxes:
        centre = [(new_box[0] + new_box[2]) // 2, (new_box[1] + new_box[3]) // 2]
        difference = abs(np.array(centre) - np.array(face_centre)).mean()
        boxes_distances.append([difference, new_box])
    if len(boxes_distances) == 0:
        return []
    boxes_distances.sort(key=lambda x: x[0])
    wanted_box = boxes_distances[0][1]
    return wanted_box

def poisson_blend(paste_image, source_image, box):
    """
    Performs Poisson blending using seamlessClone to blend paste_image into source_image.

    Args:
        paste_image (numpy.ndarray): The image to be pasted onto the source image.
        source_image (numpy.ndarray): The source image.
        box (tuple): The coordinates of the bounding box in the form (x1, y1, x2, y2).

    Returns:
        numpy.ndarray: The blended image.
    """

    src_mask = np.zeros(paste_image.shape, dtype=paste_image.dtype)
    height, width = paste_image.shape[:2]

    rectangle = np.array([
        [0, 0], 
        [0, height],
        [width, height],
        [width, 0]], dtype=np.int32)
    cv2.fillPoly(src_mask, [rectangle], (255, 255, 255))

    box_centre = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
    blended = cv2.seamlessClone(paste_image, source_image, src_mask, box_centre, cv2.NORMAL_CLONE)

    return blended

def train(G, D, fixed_noise, cropped_real_face_tensor, mask, lr=0.0003, iterations=1500, lam=0.1, eval_interval=200):
    """
    Trains the generator G using adversarial loss, contextual loss, and perceptual loss.

    Args:
        G: The generator model.
        D: The discriminator model.
        fixed_noise (torch.Tensor): The fixed noise vector.
        cropped_real_face_tensor (torch.Tensor): The tensor representing the cropped real face.
        mask (torch.Tensor): The mask tensor.
        lr (float): The learning rate (default: 0.0003).
        iterations (int): The number of training iterations (default: 1500).
        lam (float): The perceptual loss factor (default: 0.1).
        eval_interval (int): The interval at which to evaluate and print progress (default: 200).

    Returns:
        list: The list of generated face images during training.
    """

    progress = []
    fixed_noise = fixed_noise.clone().requires_grad_(True)

    optimizer = optim.Adam([fixed_noise], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    t_start = time.time()

    for i in range(iterations):
        fake_face = G(fixed_noise, None)
        contextual_loss = nn.functional.l1_loss(mask * fake_face, mask * cropped_real_face_tensor)
        perceptual_loss = D(fake_face, None)[0][0]

        complete_loss = contextual_loss + lam * perceptual_loss

        optimizer.zero_grad()
        complete_loss.backward()
        optimizer.step()
        scheduler.step()

        if i % eval_interval == 0 or i == iterations - 1:
            print(f"Losses, {i} iteration:: Complete: {complete_loss:.4f}, Contextual: {contextual_loss:.4f}, Perceptual: {lam * perceptual_loss:.4f} (after x{lam}), Time: {time.time() - t_start:.2f}s")
            progress.append((cropped_real_face_tensor * mask + fake_face * (1 - mask)).cpu())

    return progress

def tensor_to_np(tensor):
    """
    Converts a Torch tensor to a normalized NumPy image.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        numpy.ndarray: The converted NumPy image.
    """
    torch_grid = torchvision.utils.make_grid(tensor.cpu(), normalize=True, padding=0)
    np_image = np.ascontiguousarray((torch_grid.permute(1, 2, 0).numpy() * 255), dtype=np.uint8)
    return np_image

def generate_inpainting_inputs(G, D, mtcnn, device, cropped_face, fixed_noise, border_factor=0.15):
    """
    Generates inputs for inpainting by obtaining a generated image, bounding boxes, and masks.

    Args:
        G: The generator model.
        D: The discriminator model.
        mtcnn: An instance of MTCNN (face detection model).
        device: The device to run the computation on.
        cropped_face (numpy.ndarray): The cropped real face image.
        fixed_noise (torch.Tensor): The fixed noise tensor.
        border_factor (float): The factor to determine the size of the border (default: 0.15).

    Returns:
        torch.Tensor, torch.Tensor, tuple, numpy.ndarray: The cropped real face tensor, mask tensor,
        generated bounding box, and resized generated face.
    """

    generated_img_tensor = G(fixed_noise, None)
    generated_img = tensor_to_np(generated_img_tensor)

    generated_boxes, generated_landmarks, generated_probs = generate_boxes_landmarks(generated_img, mtcnn, device)

    assert len(generated_boxes) >= 1, "No faces detected in the generated image"
    assert len(generated_boxes) == 1, "Two faces detected in the generated image"
    
    box_generated = generated_boxes[0]
    x1, y1, x2, y2 = box_generated
    width = x2 - x1
    height = y2 - y1

    tensor_transform = transforms.ToTensor()

    border_width = int(width * border_factor)
    border_height = int(height * border_factor)

    mask = torch.zeros((1, 3, 1024, 1024)).to(device)
    mask[:, :, y1:y2, x1:x2] = 1
    mask[:, :, y1 + border_height:y2 - border_height, x1 + border_width:x2 - border_width] = 0

    resized_face = cv2.resize(cropped_face.copy(), (width, height))
    cropped_real_face_tensor = torch.zeros((1, 3, 1024, 1024)).to(device)
    cropped_real_face_tensor[:, :, y1:y2, x1:x2] = tensor_transform(resized_face).unsqueeze(dim=0).to(device)

    generated_face = generated_img[y1:y2, x1:x2]
    resized_face = cv2.resize(generated_face, (cropped_face.shape[1], cropped_face.shape[0]))
    border_width = int(resized_face.shape[1] * border_factor)
    border_height = int(resized_face.shape[0] * border_factor)
    resized_face = resized_face[border_height:-border_height, border_width:-border_width]

    return cropped_real_face_tensor, mask, box_generated, resized_face
    
def inpaint(G, D, mtcnn, device, cropped_face, fixed_noise, cropped_real_face_tensor, mask, box_generated, lr=0.0003, iterations=1500, lam=0.1, eval_interval=200, border_factor=0.15):
    """
    Performs inpainting by training the generator and generating inpainted faces.

    Args:
        G: The generator model.
        D: The discriminator model.
        mtcnn: An instance of MTCNN (face detection model).
        device: The device to run the computation on.
        cropped_face (numpy.ndarray): The cropped real face image.
        fixed_noise (torch.Tensor): The fixed noise tensor.
        cropped_real_face_tensor (torch.Tensor): The tensor representing the cropped real face.
        mask (torch.Tensor): The mask tensor.
        box_generated (tuple): The generated bounding box coordinates.
        lr (float): The learning rate for training (default: 0.0003).
        iterations (int): The number of training iterations (default: 1500).
        lam (float): The perceptual loss factor for training (default: 0.1).
        eval_interval (int): The interval at which to evaluate and print progress during training (default: 200).
        border_factor (float): The factor to determine the size of the border for inpainted faces (default: 0.15).

    Returns:
        list: A list of inpainted faces.
    """

    progress = train(G, D, fixed_noise, cropped_real_face_tensor, mask, lr=lr, iterations=iterations, lam=lam, eval_interval=eval_interval)

    inpainted_faces = []
    x1, y1, x2, y2 = box_generated
    for inpainted_face in progress:
        inpainted_face = tensor_to_np(inpainted_face[:, :, y1:y2, x1:x2])
        inpainted_face = cv2.resize(inpainted_face, (cropped_face.shape[1], cropped_face.shape[0]))

        border_width = int(inpainted_face.shape[1] * border_factor)
        border_height = int(inpainted_face.shape[0] * border_factor)
        inpainted_face = inpainted_face[border_height:-border_height, border_width:-border_width]
        inpainted_faces.append(inpainted_face)

    return inpainted_faces

def visualize_progress(original_img_padded, box, landmark, face_centre, rotate_angle, rotated_img, rotated_box, pad_width, inpainted_faces, border_factor):
    """
    Visualizes the progress of inpainting by drawing annotations and blending inpainted faces.

    Args:
        original_img_padded (numpy.ndarray): The original image padded with borders.
        box (tuple): The original face bounding box coordinates.
        landmark (list): The landmarks of the face.
        face_centre (tuple): The coordinates of the face center.
        rotate_angle (float): The rotation angle in degrees.
        rotated_img (numpy.ndarray): The rotated image.
        rotated_box (tuple): The rotated face bounding box coordinates.
        pad_width (int): The width of the padding.
        inpainted_faces (list): A list of inpainted faces.
        border_factor (float): The factor to determine the size of the border for obfuscation.

    Returns:
        list: A list of annotated and blended images.
    """

    images_to_save = []

    # 1. Draw the original face bounding box
    annotated_faces_img = original_img_padded.copy()
    cv2.rectangle(annotated_faces_img, box[:2], box[2:], (255, 0, 0), 2)
    images_to_save.append(annotated_faces_img.copy())

    # 2. Draw landmarks (eyes)
    left_eye, right_eye = landmark[0], landmark[1]
    cv2.circle(annotated_faces_img, left_eye, 3, (0, 255, 255), -1)
    images_to_save.append(annotated_faces_img.copy())
    cv2.circle(annotated_faces_img, right_eye, 3, (0, 255, 255), -1)
    images_to_save.append(annotated_faces_img.copy())

    # 3. Create and draw the third point to form a triangle
    if left_eye[1] > right_eye[1]:
        third_point = (right_eye[0], left_eye[1])
    else:
        third_point = (left_eye[0], right_eye[1])
    cv2.circle(annotated_faces_img, third_point, 3, (0, 255, 255), -1)
    images_to_save.append(annotated_faces_img.copy())

    # 4. Draw the triangle using the landmarks
    for pair in [[left_eye, right_eye], [left_eye, third_point], [right_eye, third_point]]:
        cv2.line(annotated_faces_img, pair[0], pair[1], (0, 255, 0), 1)
    images_to_save.append(annotated_faces_img.copy())

    # 5. Rotate the image and add rotation angle text
    rotated_img_annotated = np.array(Image.fromarray(annotated_faces_img).rotate(rotate_angle, center=face_centre, expand=False))
    cv2.putText(rotated_img_annotated, f"Rotating by {rotate_angle:.2f}° anti-clockwise", (pad_width + 10, pad_width + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    images_to_save.append(rotated_img_annotated.copy())

    # 6. Draw the rotated face bounding box on the annotated image
    cv2.rectangle(rotated_img_annotated, rotated_box[:2], rotated_box[2:], (0, 255, 0), 2)
    images_to_save.append(rotated_img_annotated.copy())

    # 7. Draw the face bounding box on the rotated image
    rotated_img_annotated = rotated_img.copy()
    cv2.rectangle(rotated_img_annotated, rotated_box[:2], rotated_box[2:], (0, 255, 0), 2)
    images_to_save.append(rotated_img_annotated.copy())

    # 8. Perform facial obfuscation
    x1, y1, x2, y2 = rotated_box
    width = x2 - x1
    height = y2 - y1
    width_border = int(width * border_factor)
    height_border = int(height * border_factor)
    rotated_img_obscured = rotated_img_annotated.copy()
    cv2.rectangle(rotated_img_obscured, [x1 + width_border, y1 + height_border], [x2 - width_border - 1, y2 - height_border - 1], (255, 255, 255), -1)
    images_to_save.append(rotated_img_obscured.copy())

    # 9. Rotate the obscured image back to its original orientation
    unrotated_img_annotated = np.array(Image.fromarray(rotated_img_obscured).rotate(-rotate_angle, center=face_centre, expand=False))
    images_to_save.append(unrotated_img_annotated.copy())

    # 10. Place the raw generated face onto the image
    rotated_img_annotated_raw = rotated_img_annotated.copy()
    start_x, start_y = rotated_box[1] + height_border, rotated_box[0] + width_border
    rotated_img_annotated_raw[start_x:start_x + inpainted_faces[0].shape[0], start_y:start_y + inpainted_faces[0].shape[1]] = inpainted_faces[0] # naive replacement
    unrotated_img_annotated_raw = np.array(Image.fromarray(rotated_img_annotated_raw).rotate(-rotate_angle, center=face_centre, expand=False))
    images_to_save.append(unrotated_img_annotated_raw.copy())

    # 11. Perform poisson blending and show the progress
    for inpainted_face in inpainted_faces:
        rotated_img_blended = rotated_img_annotated.copy()
        rotated_img_blended = poisson_blend(inpainted_face, rotated_img_blended, rotated_box)
        unrotated_img_blended = np.array(Image.fromarray(rotated_img_blended).rotate(-rotate_angle, center=face_centre, expand=False))
        images_to_save.append(unrotated_img_blended.copy())

    return images_to_save
