from PIL import Image

# 图像文件的路径
image_path = '/code/ResUnet/SIIM-ACR-Pneumothorax-Seg-XR/valid/output_1/1.2.276.0.7230010.3.1.4.8323329.305.1517875162.307235.png'

# 读取图像
image = Image.open(image_path)

# 输出图像的尺寸（宽度, 高度）
print("Image Shape:", image.size)

# 关闭图像文件
image.close()
