import os
import pickle
import random

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "Apple___Apple_scab": "apple scab disease leaf",
    "Apple___Black_rot": "apple black rot disease leaf",  
    "Apple___Cedar_apple_rust": "cedar apple rust disease leaf",
    "Apple___healthy": "healthy apple leaf",
    "Blueberry___healthy": "healthy blueberry leaf",
    "Cherry___healthy": "healthy cherry leaf",
    "Cherry___Powdery_mildew": "cherry powdery mildew disease leaf",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "corn cercospora spot disease leaf",
    "Corn___Common_rust": "corn common rust disease leaf",
    "Corn___healthy": "healthy corn leaf",
    "Corn___Northern_Leaf_Blight": "corn northern blight disease leaf",
    "Grape___Black_rot": "grape black rot",
    "Grape___Esca_(Black_Measles)": "grape black measles disease leaf",
    "Grape___healthy": "healthy grape leaf",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape blight disease leaf",
    "Orange___Haunglongbing_(Citrus_greening)": "orange citrus greening disease leaf",
    "Peach___Bacterial_spot": "peach bacterial spot disease leaf",
    "Peach___healthy": "healthy peach leaf",
    "Pepper,_bell___Bacterial_spot": "bell pepper bacterial spot disease leaf",
    "Pepper,_bell___healthy": "healthy bell pepper leaf",
    "Potato___Early_blight": "potato early blight disease leaf",
    "Potato___healthy": "healthy potato leaf",
    "Potato___Late_blight": "potato late blight disease leaf",
    "Raspberry___healthy": "healthy raspberry leaf",
    "Soybean___healthy": "healthy soybean leaf",
    "Squash___Powdery_mildew": "squash powdery mildew disease leaf",
    "Strawberry___healthy": "healthy strawberry leaf",
    "Strawberry___Leaf_scorch": "strawberry scorch disease leaf",
    "Tomato___Bacterial_spot": "tomato bacterial spot disease leaf",
    "Tomato___Early_blight": "tomato early blight disease leaf",
    "Tomato___healthy": "healthy tomato leaf",
    "Tomato___Late_blight": "tomato blight disease leaf",
    "Tomato___Leaf_Mold": "tomato mold disease leaf",
    "Tomato___Septoria_leaf_spot": "tomato septoria spot disease leaf",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato two spotted spider mite disease leaf",
    "Tomato___Target_Spot": "tomato target spot disease leaf",
    "Tomato___Tomato_mosaic_virus": "tomato mosaic virus disease leaf",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato yellow curl virus disease leaf",
}

@DATASET_REGISTRY.register()
class PlantVillage(DatasetBase):

    dataset_dir = "plant_village"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_Plant_Disease.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_and_split_data(self.image_dir,new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        print("Number of shots = ",num_shots)
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.6, p_val=0.2, ignored=[], new_cnames=None):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     dog/
        #     cat/
        #     horse/
        # =============
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train : n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val :], label, category))

        return train, val, test
