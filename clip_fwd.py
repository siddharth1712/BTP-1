import torch
import clip
from PIL import Image
import os
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

healthy_path = "plant_images/healthy"
infected_path = "plant_images/infected"

healthy_images=os.listdir(healthy_path)
infected_images=os.listdir(infected_path)

pred=[]
actual=[]

labels=["an infected leaf","a healthy leaf"]
caption=['A photo of '+label for label in labels]

for j in infected_images:
    image = preprocess(Image.open(infected_path+"/"+j)).unsqueeze(0).to(device)
    text = clip.tokenize(caption).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    pred.append(np.argmax(probs))
    actual.append(0)

for i in healthy_images:
    image = preprocess(Image.open(healthy_path+"/"+i)).unsqueeze(0).to(device)
    text = clip.tokenize(caption).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    pred.append(np.argmax(probs))
    actual.append(1)

diff=np.array(pred)-np.array(actual)
diff=np.array(diff)

acc=np.count_nonzero(diff)
print("Accuracy = "+str(1-(acc/len(diff))))