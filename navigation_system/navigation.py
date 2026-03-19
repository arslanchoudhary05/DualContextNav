from ultralytics import YOLO
import cv2
import numpy as np

indoor_model = YOLO("weights/indoor_best.pt")
outdoor_model = YOLO("weights/outdoor_best.pt")

def classify_environment(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    edges = cv2.Canny(gray,100,200)
    edge_density = np.sum(edges>0)/edges.size
    color_var = np.var(img/255.0)
    score = 0.4*brightness + 0.3*edge_density + 0.3*color_var
    return "indoor" if score < 60 else "outdoor"

def get_position(box, w):
    cx = (box[0]+box[2])/2
    if cx < w/3:
        return "left"
    elif cx > 2*w/3:
        return "right"
    else:
        return "front"

def get_distance(box, h):
    ratio = (box[3]-box[1])/h
    if ratio > 0.5:
        return "near"
    elif ratio > 0.25:
        return "medium"
    else:
        return "far"

def select_target(objects):
    candidates = [o for o in objects if o["pos"]=="front" and o["dist"]!="far"]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x["height"])

def decision(target, env):
    if target is None:
        return "No obstacle"
    if target["dist"]=="medium":
        return f"Warning: {target['name']} ahead"
    if target["dist"]=="near":
        if env=="indoor":
            return f"{target['name']} very close. Adjust path"
        else:
            return f"ALERT: {target['name']} very close"

def run_navigation(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found")
        return
    h,w,_ = img.shape
    env = classify_environment(img)
    model = indoor_model if env=="indoor" else outdoor_model
    results = model(img)[0]
    objects = []
    for box,cls in zip(results.boxes.xyxy,results.boxes.cls):
        box = box.cpu().numpy().tolist()
        cls = int(cls.cpu().numpy())
        obj = model.names[cls]
        pos = get_position(box,w)
        dist = get_distance(box,h)
        objects.append({
            "name":obj,
            "pos":pos,
            "dist":dist,
            "height":box[3]-box[1]
        })
    target = select_target(objects)
    action = decision(target, env)
    print("Environment:",env)
    print("Objects:",len(objects))
    print("Decision:",action)

if __name__ == "__main__":
    run_navigation("test.jpg")
