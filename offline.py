import glob
import os
import pickle
from PIL import Image
from tqdm import tqdm
from feature_extractor import FeatureExtractor
fe = FeatureExtractor()

def gen_future(img_path):
    # for keras multiprocessing,init must be in func
    # from feature_extractor import FeatureExtractor 
    # fe = FeatureExtractor()
    img = Image.open(img_path)  # PIL image
    feature = fe.extract(img)
    feature_path = 'static/feature/' + os.path.splitext(os.path.basename(img_path))[0] + '.pkl'
    pickle.dump(feature, open(feature_path, 'wb'))

# normal
if __name__ == "__main__":
    for img_path in tqdm(sorted(glob.glob('static/img/*.jpg'))):
        gen_future(img_path)


# multiprocessing
# import multiprocessing
# cores = multiprocessing.cpu_count()

# if __name__ == "__main__":
#     pool = multiprocessing.Pool(processes=cores)
#     img_filter = [img_path for img_path in sorted(glob.glob('static/test/*.jpg'))]
#     for _ in tqdm(pool.imap_unordered(gen_future, img_filter), total=len(img_filter)):
#         pass
#     # pool.map(gen_future,img_filter)
