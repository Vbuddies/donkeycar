import cv2
import numpy as np
import logging
from donkeycar.config import Config


logger = logging.getLogger(__name__)

#
# HACK: workaround for imgaug bug; mock our implementation
# TODO: remove this when https://github.com/autorope/donkeycar/issues/970
#       is addressed.
#
try:
    import imgaug.augmenters as iaa

    class Augmentations(object):
        """
        Some ready to use image augumentations.
        """

        @classmethod
        def crop(cls, left, right, top, bottom, keep_size=False):
            """
            The image augumentation sequence.
            Crops based on a region of interest among other things.
            left, right, top & bottom are the number of pixels to crop.
            """
            augmentation = iaa.Crop(px=(top, right, bottom, left),
                                    keep_size=keep_size)
            return augmentation

        @classmethod
        def trapezoidal_mask(cls, lower_left, lower_right, upper_left,
                             upper_right, min_y, max_y):
            """
            Uses a binary mask to generate a trapezoidal region of interest.
            Especially useful in filtering out uninteresting features from an
            input image.
            """
            def _transform_images(images, random_state, parents, hooks):
                # Transform a batch of images
                transformed = []
                mask = None
                for image in images:
                    if mask is None:
                        mask = np.zeros(image.shape, dtype=np.int32)
                        # # # # # # # # # # # # #
                        #       ul     ur          min_y
                        #
                        #
                        #
                        #    ll             lr     max_y
                        points = [
                            [upper_left, min_y],
                            [upper_right, min_y],
                            [lower_right, max_y],
                            [lower_left, max_y]
                        ]
                        cv2.fillConvexPoly(mask,
                                           np.array(points, dtype=np.int32),
                                           [255, 255, 255])
                        mask = np.asarray(mask, dtype='bool')

                    masked = np.multiply(image, mask)
                    transformed.append(masked)

                return transformed

            def _transform_keypoints(keypoints_on_images, random_state,
                                     parents, hooks):
                # No-op
                return keypoints_on_images

            augmentation = iaa.Lambda(func_images=_transform_images,
                                      func_keypoints=_transform_keypoints)
            return augmentation
        
        @classmethod
        def segmentation(cls):
            """Image Augmentation to enact road segmentation"""


            def _edge_detection(frame):
                '''want to identify road boundaries'''

                import numpy as np
                import cv2
                from skimage.segmentation import slic
                import pdb

                # pre defined tools that we don't want to write over every time this runs
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                kernel_dilate = np.ones((3, 3), np.uint8)
                tophalf = np.zeros((50, 160))

                
                # load image
                img_arr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # read in as greyscale
                img_arr = img_arr[50:]  # cut off top half of picture, eliminates background destractions, smaller image size to improve performance

                # identify edges in image
                cl1 = clahe.apply(img_arr)
                edges_cl1 = cv2.Canny(cl1, 100, 200)

                # dilate to create lines
                img_dilate_1 = cv2.dilate(edges_cl1, kernel_dilate, iterations=1)

                # identify connected components
                analysis = cv2.connectedComponentsWithStats(img_dilate_1, 8, cv2.CV_32S)
                (totalLabels, label_ids, values, centroid) = analysis

                # create a mask to find larger components 
                output = np.zeros_like(img_dilate_1)
                for i in range(1, totalLabels):
                    area = values[i, cv2.CC_STAT_HEIGHT] * values[i, cv2.CC_STAT_WIDTH]
                    componentMask = (label_ids == i).astype("uint8") * 255

                    if area > 1000:
                        # Creating the Final output mask
                        output = cv2.bitwise_or(output, componentMask)

                # put on top of image as all zeros
                final = np.vstack((tophalf, output))

                # extend shape for training purposes
                final2 = cv2.merge([final, final, final])

                return final2

            

            def _custom(images, random_state, parents, hooks):
                transformed = []

                for img in images:
                    transformed.append(_edge_detection(img))
                
                return transformed

            def _transform_keypoints(keypoints_on_images, random_state, parents, hooks):
                return keypoints_on_images

            augmentation = iaa.Lambda(func_images=_custom, func_keypoints=_transform_keypoints)
            return augmentation
    

    class ImageAugmentation:
        def __init__(self, cfg, key):
            aug_list = getattr(cfg, key, [])
            augmentations = \
                [ImageAugmentation.create(a, cfg) for a in aug_list]
            self.augmentations = iaa.Sequential(augmentations)

        @classmethod
        def create(cls, aug_type: str, config: Config) -> iaa.meta.Augmenter:
            """ Augmenatition factory. Cropping and trapezoidal mask are
                transfomations which should be applied in training, validation
                and inference. Multiply, Blur and similar are augmentations
                which should be used only in training. """

            if aug_type == 'CROP':
                logger.info(f'Creating augmentation {aug_type} with ROI_CROP '
                            f'L: {config.ROI_CROP_LEFT}, '
                            f'R: {config.ROI_CROP_RIGHT}, '
                            f'B: {config.ROI_CROP_BOTTOM}, '
                            f'T: {config.ROI_CROP_TOP}')

                return Augmentations.crop(left=config.ROI_CROP_LEFT,
                                          right=config.ROI_CROP_RIGHT,
                                          bottom=config.ROI_CROP_BOTTOM,
                                          top=config.ROI_CROP_TOP,
                                          keep_size=True)
            elif aug_type == 'TRAPEZE':
                logger.info(f'Creating augmentation {aug_type}')
                return Augmentations.trapezoidal_mask(
                            lower_left=config.ROI_TRAPEZE_LL,
                            lower_right=config.ROI_TRAPEZE_LR,
                            upper_left=config.ROI_TRAPEZE_UL,
                            upper_right=config.ROI_TRAPEZE_UR,
                            min_y=config.ROI_TRAPEZE_MIN_Y,
                            max_y=config.ROI_TRAPEZE_MAX_Y)

            elif aug_type == 'SEGMENTATION':
                logger.info(f'Custom Augmentation {aug_type}')
                return Augmentations.segmentation()

            elif aug_type == 'MULTIPLY':
                interval = getattr(config, 'AUG_MULTIPLY_RANGE', (0.5, 1.5))
                logger.info(f'Creating augmentation {aug_type} {interval}')
                return iaa.Multiply(interval)

            elif aug_type == 'BLUR':
                interval = getattr(config, 'AUG_BLUR_RANGE', (0.0, 3.0))
                logger.info(f'Creating augmentation {aug_type} {interval}')
                return iaa.GaussianBlur(sigma=interval)

        # Parts interface
        def run(self, img_arr):
            aug_img_arr = self.augmentations.augment_image(img_arr)
            return aug_img_arr

except ImportError:

    #
    # mock implementation
    #
    class ImageAugmentation:
        def __init__(self, cfg, key):
            aug_list = getattr(cfg, key, [])
            for aug in aug_list:
                logger.warn(
                    'Augmentation library could not load.  '
                    f'Augmentation {aug} will be ignored')

        def run(self, img_arr):
            return img_arr
