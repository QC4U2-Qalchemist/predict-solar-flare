import numpy as np
import cv2
from PIL import Image, ImageStat
from skimage import feature, filters
import matplotlib.pyplot as plt


class FeaturesExtractor:
    def __init__(self, gauss_thresh=140, is_show_imgs=False, is_save_circumferential_denoising_npy=True):
        self.is_show_images = is_show_imgs
        self.is_save_circumferential_denoising_npy = is_save_circumferential_denoising_npy
        self.gauss_thresh = gauss_thresh
        self.frames = []
        self.features = {}
        self.features['complexities in orignal frame'] = []
        self.features['complexities in strength Gauss'] = []
        self.features['num of active regions'] = []
        self.features['AR complexity avg.'] = []
        self.features['AR complexity total'] = []

        self.pause = False
        self.circumferential_denoising = []

        #self.complexities = []

    def save_features(self, out_filepath):
        np.save(out_filepath, self.features)

    def load_features(self, in_filepath):
        self.features = np.load(in_filepath, allow_pickle=True).item()

    def plot(self, label):
        fig = plt.figure()
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        axs = []

        # �O���t�̐��͓����ʐ��{���z�t���A�̔����L��
        num_of_plot = len(self.features) + 1
        print(num_of_plot)

        # �����ʂ̎��n��f�[�^���c�ɕ��ׂĕ\��
        for i in range(num_of_plot):
            axs.append(fig.add_subplot(num_of_plot, 1, i + 1))

        # ���z�t���A�̔����L�����n��f�[�^
        axs[0].set_title('Occurrence of solar flares (1:True, 0:False)')
        axs[0].plot(range(len(label)), label, color='red')

        for i, (key, data) in enumerate(self.features.items()):
            print(key,len(data))
            axs[i+1].set_title(key)
            axs[i+1].plot(range(len(label)), data)

        plt.show()

    def hconcat_resize_min(self, im_list, interpolation=cv2.INTER_CUBIC):
        h_min = min(im.shape[0] for im in im_list)
        im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                          for im in im_list]
        return cv2.hconcat(im_list_resize)

    def append(self, image):
        self.frames.append(image)

        # ���ꋭ�x��臒l�𒴂���K�E�X�̃s�N�Z��
        high_strength_gauss = np.where(image >= self.gauss_thresh, image, 0)

        # �~�������̃m�C�Y���}�X�N����
        masked_frame = self.mask_outside_circle(high_strength_gauss)

        # �I���W�i�� ROS Magnetogram�̕��G�x
        self.features['complexities in orignal frame'].append(self.get_complexity(image))

        # ROS Magnetogram�̎��ꋭ�x�̍����̈�̕��G�x
        self.features['complexities in strength Gauss'].append(self.get_complexity(masked_frame))

        # Active regins�̒��o
        active_regions = self.get_active_regions(image)

        # Active regins��
        self.features['num of active regions'].append(len(active_regions))
    

        ar_complexity_total = 0
        for active_region in active_regions:
            # AR�̕��G�x
            ar_complexity_total += self.get_complexity(active_region)

        # AR���G�x�̕��ϒl
        ar_complexity = ar_complexity_total / len(active_regions)
        self.features['AR complexity avg.'].append(ar_complexity)
        self.features['AR complexity total'].append(ar_complexity_total)

        denoised = self.get_circumferential_denoising(image)
        self.circumferential_denoising.append(denoised)
        #cv2.imshow('denoised',denoised)
        #cv2.imshow('image',image)
        #cv2.waitKey(1)

    def find_min_max_coordinates(self, image):
        # 0�łȂ��s�N�Z���̍��W���擾
        non_zero_coords = cv2.findNonZero(image)
        
        if non_zero_coords is not None:
            # X����Y���ō��W�𕪊�
            x_coords = non_zero_coords[:,:,0]
            y_coords = non_zero_coords[:,:,1]
            
            # X����Y���̍ŏ��l�ƍő�l�����߂�
            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)
            
            return (min_x, max_x, min_y, max_y)
        else:
            return None

    def mask_outside_circle(self, image, circumference_width=8):
        # 0�łȂ��s�N�Z���̍��W�̍ŏ��l�ƍő�l�����߂�
        bounds = self.find_min_max_coordinates(image)
        if bounds is None:
            return None
        
        min_x, max_x, min_y, max_y = bounds
        
        # �~�̒��S���W���v�Z
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # �~�̔��a���v�Z
        radius = max(max_x - center_x, max_y - center_y) - circumference_width
        
        # �o�͗p�̉摜������
        masked_image = np.zeros_like(image)
        
        # �}�X�N����
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) > radius:
                    masked_image[y, x] = 0
                else:
                    masked_image[y, x] = image[y, x]
        
        return masked_image

    def get_complexity(self, image):
        scaled_image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # Canny�G�b�W���o��K�p
        edges = cv2.Canny(scaled_image, 100, 200)

        # �G�b�W�̃s�N�Z�������v�Z
        edges_pixels = np.sum(edges == 255)

        # �S�s�N�Z����
        total_pixels = image.size

        # ���G�x�̌v�Z�i�G�b�W�̊����j
        complexity = edges_pixels / total_pixels

        return complexity

    def get_resolution(self, image):
        return image.shape[0] * image.shape[1]

    def get_brightness(self, image):
        img = Image.fromarray(image)
        stat = ImageStat.Stat(img)
        return stat.mean[0]

    def get_contrast(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return img_gray.std()

    def get_subject_clarity(self, image):
        scaled_image = np.clip(image * 255, 0, 255).astype(np.uint8)
        edges = cv2.Canny(scaled_image, 100, 200)
        return np.mean(edges)

    def get_background_complexity(self, image):
        scaled_image = np.clip(image * 255, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
        entropy = filters.rank.entropy(gray, np.ones((5, 5)))
        return np.mean(entropy)

    def get_motion_blur(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return var

    def get_active_regions(self, image, rect_size=(20,20)):

        active_regions = []

        color = cv2.cvtColor(np.clip(image * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        at_rects = np.zeros(color.shape)

        # ���ꋭ�x��gauss_thresh�K�E�X�ȉ��̃s�N�Z��
        high_strength_gauss = np.where(image >= self.gauss_thresh, image, 0)

        # �~�������̃m�C�Y���}�X�N����
        masked_high_strength_gauss = self.mask_outside_circle(high_strength_gauss)

        # �֊s���o
        scaled_masked_high_strength_gauss = np.clip(masked_high_strength_gauss * 255, 0, 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(scaled_masked_high_strength_gauss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # �P�_�݂̗̂֊s�͔r��
            if cv2.contourArea(contour) > 1:
                # �֊s�̕`��
                cv2.drawContours(at_rects, [contour], -1, (255, 255, 255), 20)


        # ���ꋭ�x��-gauss_thresh�K�E�X�ȉ��̃s�N�Z��
        low_strength_gauss = np.where(-image >= self.gauss_thresh, -image, 0)

        # �~�������̃m�C�Y���}�X�N����
        masked_low_strength_gauss = self.mask_outside_circle(low_strength_gauss)

        # �֊s���o
        scaled_masked_low_strength_gauss = np.clip(masked_low_strength_gauss * 255, 0, 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(scaled_masked_low_strength_gauss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # �P�_�݂̗̂֊s�͔r��
            if cv2.contourArea(contour) > 1:
                # �֊s�̕`��
                cv2.drawContours(at_rects, [contour], -1, (255, 255, 255), 20)

        # �O���[�X�P�[���ɕϊ�
        gray = cv2.cvtColor(np.clip(at_rects * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # ��l������
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # �������֊s�݂̂�ێ�
        min_area = 500*500  # �ŏ��ʐς�ݒ�
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) < min_area]

        for contour in contours:
            # �֊s�̕`��
            cv2.drawContours(at_rects, [contour], -1, (255, 0, 255), 1)

            # �ŏ���`�̕`��
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(color, (x, y), (x + w, y + h), (255, 0, 255), 2)

            #cv2.imshow('part',image[y:y+h, x:x+w])
            #cv2.waitKey(0)
            active_regions.append(image[y:y+h, x:x+w])


        if self.is_show_images:
            for i, active_regions in enumerate(active_regions):
                cv2.imshow('Active Regions' + str(i), active_regions)

            cv2.imshow('Active Regions', color)
            cv2.imshow('org',self.hconcat_resize_min([image, high_strength_gauss, masked_high_strength_gauss]))
            self.key_handler(1)  # You might want to adjust this depending on the context

        return active_regions

    def save_circumferential_denoising(self, out_filepath):
        np.save(out_filepath,np.array(self.circumferential_denoising).transpose(1, 2, 0))
        return 

    def get_circumferential_denoising(self, image, rect_size=(20,20)):
        denoised = self.mask_outside_circle(image[:, ::-1]).copy()
        return denoised


    def key_handler(self, wait_in_ms):

        key = cv2.waitKey(wait_in_ms)
        if key == ord(' '):  # �X�y�[�X�L�[�ňꎞ��~�E�Đ�
            self.pause = not self.pause
        while self.pause:
            key = cv2.waitKey(100)
            if key == ord(' '):
                self.pause = False
                break
