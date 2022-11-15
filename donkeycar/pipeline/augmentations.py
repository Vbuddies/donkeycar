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
                kernel_dilate = np.ones((4, 4), np.uint8)
                kernel_erode = np.ones((6, 6), np.uint8)
                tophalf = np.zeros((50, 160))

                
                # load image
                img_arr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # read in as greyscale
                img_arr = img_arr[50:]  # cut off top half of picture, eliminates background destractions, smaller image size to improve performance
                
                #### strategy ####
                # want to use edge detection to find lines
                # use contrasting to better highlight lines
                # bitwise and 2 different methods of contrasting to find road lines
                # dilate and erode edges to create mask for original image
                # stack 0s on top of half image to create full image again

                # first contrast image usually does better with lighter images
                equ = cv2.equalizeHist(img_arr)
                img_arr_blur = cv2.blur(equ, (3, 3))
                edges_blur = cv2.Canny(img_arr_blur, 100, 200)


                # second contrast image usually does better with darker images
                cl1 = clahe.apply(img_arr)
                edges_cl1 = cv2.Canny(cl1, 100, 200)


                # where both contrasts agree is mostly road edge
                edges_and = cv2.bitwise_and(edges_cl1, edges_blur)


                # dilate and erode to eliminate smaller fragments that are likely not road
                img_dilate_1 = cv2.dilate(edges_and, kernel_dilate, iterations=1)
                img_erode = cv2.erode(img_dilate_1, kernel_erode, iterations=1)
                img_dilate_2 = cv2.dilate(img_erode, kernel_dilate, iterations=1)


                # put in top of image as all zeros
                final = np.vstack((tophalf, img_dilate_2))

                final2 = cv2.merge([final, final, final])

                # final is a black and white image where white is estimated road edge
                # could be used as mask for original image or as the image itself
                return final2


                # use these to show filtering results
                # cv2.imshow('final', final)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            

            def _custom(images, random_state, parents, hooks):
                import cv2
                import numpy as np

                transformed = []

                kernel_dilate = np.ones((4, 4), np.uint8)
                kernel_erode = np.ones((6, 6), np.uint8)
                tophalf = np.zeros((50, 160, 3))

                for img in images:
                    if False:
                        gray = iaa.Grayscale(alpha=1.0)
                        img_gray = gray.augment_image(img)
                        img = img_gray[50:]

                        equalizeHist = iaa.HistogramEqualization()
                        equ = equalizeHist.augment_image(img)

                        # blur
                        blur = iaa.GaussianBlur(sigma=(3, 3))
                        img_blur = blur.augment_image(equ)

                        # edgeblur
                        canny = iaa.Canny(hysteresis_thresholds=(100,200), colorizer=iaa.RandomColorsBinaryImageColorizer(color_true=255, color_false=0))
                        edges_blur = canny.augment_image(img_blur)


                        #second contrast image
                        clahe = iaa.CLAHE(clip_limit=2, tile_grid_size_px=(8,8))
                        cl1 = clahe.augment_image(img)
                        edges_cl1 = canny.augment_image(cl1)


                        # where contrast mostly agree is the road
                        edges_and = cv2.bitwise_and(edges_cl1, edges_blur)

                        # dilate and erode to eliminate smaller fragments that are likely not road
                        img_dilate_1 = cv2.dilate(edges_and, kernel_dilate, iterations=1)
                        img_erode = cv2.erode(img_dilate_1, kernel_erode, iterations=1)
                        img_dilate_2 = cv2.dilate(img_erode, kernel_dilate, iterations=1)


                        # put in top of image as all zeros
                        final = np.vstack((tophalf, edges_and))


                        # Use this for Justin's Code
                        # img = _edge_detection(img)
                        
                        transformed.append(final)
                    else:
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
