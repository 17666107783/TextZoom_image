import lmdb
import numpy as np
import cv2
import pdb
import six
from PIL import Image
import utilities

def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im

def read_lmdb(lmdb_file, savepath):
    lmdb_env = lmdb.open(
            lmdb_file, 
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
    if not lmdb_env:
        print('cannot creat lmdb from %s' % (lmdb_file))
        sys.exit(0)
    with lmdb_env.begin(write=False) as txn:
        nSamples = int(txn.get(b'num-samples'))
        print(nSamples)
        for index in range(1, nSamples+1):
            img_HR_key = b'image_hr-%09d' % index 
            img_lr_key = b'image_lr-%09d' % index
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')

            try:
                img_HR.save(savepath + str(index) + '_img_HR.jpg', quality=95)
                img_lr.save(savepath + str(index) + '_img_LR.jpg', quality=95)
            except IOError:
                print('Corrupted image for %d' % index)
                return 
            label_key = b'label-%09d' % index
            word = str(txn.get(label_key).decode())
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()))
            # pdb.set_trace()


def main():
    savepath = 'train2_img/'
    utilities.mkdir(savepath)
    lmdb_file = 'train2'
    read_lmdb(lmdb_file, savepath)

if __name__ == '__main__':
    main()