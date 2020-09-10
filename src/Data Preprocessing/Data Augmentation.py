import os, time, numpy, random
from cv2 import cv2


class ImageProcessor():
    def __init__(self, root_path, export_path):
        self._rootPath = root_path
        #print("__init__")
        self._exportPath = export_path  
        if not os.path.exists(export_path):
            os.mkdir(export_path) 
        

    def processes(self,path):
        #find the image path to process
        ori_path = path
        for file_name in os.listdir(self._rootPath):
            file_path = os.path.join(self._rootPath,file_name)
            export_path = os.path.join(ori_path,file_name)
            self._exportPath = export_path
            print("***"+export_path)
            print("***"+self._exportPath)
            if not os.path.exists(export_path):
                os.mkdir(export_path)
            for image_name in os.listdir(file_path):
                image_path = os.path.join(file_path,image_name)
                if os.path.isfile(image_path): #if not a file, passed directly
                    title, type = image_name.split('.') 
                    if type != 'jpg' and type != 'JPG':#check whether this is an image
                        print("it's not an image:", image_name)
                        continue
                    #open image by imread
                    ori_image = cv2.imread(image_path)
                    print(image_path)
                    print("processing:", image_name)
                    #add noise (data augmentation)
                    self._multi_process(ori_image, title + '_original')
                    for new_name, cutted_image in self.cut(ori_image):
                        self._multi_process(cutted_image, title + new_name)
        
    def _multi_process(self, ori_image, image_name):
        #print(ori_image)
        down_sized = self.resize(ori_image)
        self._save('downSize', image_name, down_sized)
        #add Gaussian noise 
        gasuss_noise = self.gasuss_noise(down_sized)
        self._save('gasussNoise', image_name, gasuss_noise)
        #add Gaussian blur 
        gasuss_blur = self.gasuss_blur(down_sized)
        self._save('gasussBlur', image_name, gasuss_blur)
        #rotate original image
        rotated = self.rotate(down_sized)
        self._save("rotate", image_name, rotated)
        #flip original image
        flipped = self.flip(down_sized)
        self._save("flip", image_name, flipped)
        #Affine and project the image
        img_aff, img_per = self.transform(down_sized)
        self._save("affine", image_name, img_aff)
        self._save("perspective", image_name, img_per)
        salt_pepper = self.sp_noise(down_sized)
        self._save("saltPepper", image_name, salt_pepper)

    def _save(self, process, image_name, image):
        label = time.strftime("%Y%m%d%H%M%S", time.localtime())
        # save to the path, and make the path_name
        savename = image_name + '_' + process + '_' + label + ".jpg"
        cv2.imwrite(os.path.join(self._exportPath, savename), image)
        

    def resize(self, image):
        #resized to 256*256*3 (3 is for RGB)
        resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        return resized

    def cut(self, image):
        rows, cols = image.shape[:2]
        y = int(rows / 2)
        x = int(cols / 2)
        yield '_upLeft', image[0:y, 0:x]
        yield '_upRight', image[0:y, x:cols]
        yield '_downLeft', image[y:rows, 0:x]
        yield '_downRight', image[y:rows, x:cols]

    def gasuss_noise(self, image, mean=0, variance=0.002):
        #Gaussian noise 
        image = numpy.array(image / 255, dtype=float)
        noise = numpy.random.normal(mean, variance**0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = numpy.clip(out, low_clip, 1.0)
        out = numpy.uint8(out * 255)
        return out

    def gasuss_blur(self, image, size=2):
        #Gaussian blur
        return cv2.GaussianBlur(image, (0, 0), size)

    def sp_noise(self, image, prob=0.03):
        rows, cols = image.shape[:2]
        thres = 1 - prob
        for i in range(cols):
            for j in range(rows):
                rdn = random.random()
                if rdn < prob:
                    image[i][j] = 0
                elif rdn > thres:
                    image[i][j] = 255
        return image

    def rotate(self, image, angle=0, center=None, scale=0.9):
        #rotate the image with "angle"
        rows, cols = image.shape[:2]
        while angle == 0:
            angle = random.randrange(-180, 180)
        if center is None:
            center = (cols / 2, rows / 2)
        #calculated the matrix
        Martrix_rot = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, Martrix_rot, (cols, rows))
        return rotated

    def flip(self, image, dirction=None):
        #flip the image
        if dirction is None:
            dirction = random.randint(-1, 1)

        flipped = cv2.flip(image, dirction)
        return flipped

    def transform(self, image):
        rows, cols = image.shape[:2]
        scale1 = (random.randrange(30, 45) / 100)
        scale2 = (random.randrange(30, 45) / 100)
        scale3 = (random.randrange(1, 5) / 10)
        scale4 = (random.randrange(1, 5) / 10)
        point1 = numpy.float32([[50, 50], [rows - 50, 50], [50, cols - 50],
                                [rows - 50, cols - 50]])
        point2 = numpy.float32([[(100 * scale1), (100 * scale2)],
                                [(rows * (1 - scale1)), (cols * scale3)],
                                [(rows * scale4), (cols * (1 - scale2))],
                                [(rows * (1 - scale3)),
                                 (cols * (1 - scale4))]])
        point1_ = point1[0:3]
        point2_ = point2[0:3]
        Matirx_aff = cv2.getAffineTransform(point1_, point2_)
        img_aff = cv2.warpAffine(image,
                                 Matirx_aff, (cols, rows),
                                 borderValue=(255, 255, 255))
        Matirx_per = cv2.getPerspectiveTransform(point1, point2)
        img_per = cv2.warpPerspective(image,
                                      Matirx_per, (cols, rows),
                                      borderValue=(255, 255, 255))
        return img_aff, img_per


if __name__ == '__main__':
    #path for the image folder
    root = r""
    #path for export image folder
    export = r""
    myproject = ImageProcessor(root,export)
    myproject.processes("")