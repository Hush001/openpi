import os
import numpy as np
import h5py
from tqdm import tqdm
import cv2
from PIL import Image
import gc

# 定义示例 prompt
example_prompt = b"pick up the ball."



def convert_images_to_binary(data):
    """
    将 uint8 格式的图像数据转换为二进制数据
    :param data: uint8 格式的图像数据
    :return: 转换后的二进制数据列表以及每个图像的二进制长度列表
    """
    binary_data_list = []
    lengths = []
    for image in tqdm(data, desc="Converting images", unit="image"):
        try:
            # 将 numpy 数组转换为 PIL Image 对象
            pil_image = Image.fromarray(image)

            # 将PIL图像转换为numpy数组
            img_array = np.array(pil_image)

            # 从RGB转换为BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # 将转换后的numpy数组再转换回PIL图像
            pil_image_bgr = Image.fromarray(img_bgr)

            # 将 PIL Image 保存为 JPEG 格式的字节流
            from io import BytesIO
            buffer = BytesIO()
            pil_image_bgr.save(buffer, format="JPEG")
            binary_data = buffer.getvalue()

            binary_data_list.append(binary_data)
            lengths.append(len(binary_data))
        except Exception as e:
            print(f"Error converting image: {e}")
    return binary_data_list, lengths


def process_image_datasets(src_file, dst_file):
    image_datasets = {
        'observations/images/cam_high': 'observations/images/cam_high',
        'observations/images/cam_left': 'observations/images/cam_left_wrist',
        'observations/images/cam_right': 'observations/images/cam_right_wrist'
    }
    for src_path, dst_path in image_datasets.items():
        if src_path in src_file:
            image_data = src_file[src_path][()]
            binary_data_list, lengths = convert_images_to_binary(image_data)
            # 确定最大长度
            max_length = max(lengths)
            # 创建合适的数据类型
            dtype = f"|S{max_length}"
            # 创建一个 numpy 数组来存储所有二进制数据
            data_arr = np.empty(len(binary_data_list), dtype=dtype)
            for i, binary_data in enumerate(binary_data_list):
                # 填充二进制数据到数组中
                data_arr[i] = binary_data.ljust(max_length, b'\x00')
            try:
                # 创建数据集
                dst_file.create_dataset(dst_path, data=data_arr, dtype=dtype)
            except Exception as e:
                print(f"Error processing dataset {src_path}: {e}")
            # 释放不再使用的变量
            del image_data, binary_data_list, lengths, data_arr
            gc.collect()


def process_single_file(input_file, output_file):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # 检查文件大小，如果文件大小为 0 或者过小，可以认为可能是损坏的文件
            file_size = os.path.getsize(input_file)
            if file_size == 0 or file_size < 1024:  # 这里设置一个简单的阈值，你可以根据实际情况调整
                print(f"Skipping potentially corrupted file: {input_file}")
                return

            # 打开输入和输出文件
            infile = h5py.File(input_file, 'r')
            outfile = h5py.File(output_file, 'w')
            try:
                # 复制 action 数据集并转换为弧度制
                if 'action' in infile:
                    action_data = infile['action'][:]
                    action_data_rad = np.radians(action_data)
                    outfile.create_dataset('action', data=action_data_rad, dtype='float32')
                    del action_data, action_data_rad
                    gc.collect()

                # 创建 base_action 数据集，初始化为 0
                if 'action' in infile:
                    action_num = infile['action'].shape[0]
                    base_action_data = np.zeros((action_num, 2), dtype='float32')
                    outfile.create_dataset('base_action', data=base_action_data, dtype='float32')
                    del base_action_data
                    gc.collect()

                # 创建 instruction 数据集，使用 vlen=bytes 类型
                outfile.create_dataset('instruction', data=example_prompt, dtype=h5py.string_dtype(encoding='utf-8'))

                # 复制 observations/effort 数据集
                if 'observations/effort' in infile:
                    effort_data = infile['observations/effort'][:]
                    outfile.create_dataset('observations/effort', data=effort_data, dtype='float32')
                    del effort_data
                    gc.collect()

                #process_image_datasets(infile, outfile)

                if 'observations/images/cam_high' in infile:
                    cam_high_data = infile['observations/images/cam_high'][:]
                    outfile.create_dataset('observations/images/cam_high', data=cam_high_data, dtype='uint8')
                    del cam_high_data
                    gc.collect()

                if 'observations/images/cam_low' in infile:
                    cam_low_data = infile['observations/images/cam_low'][:]
                    outfile.create_dataset('observations/images/cam_low', data=cam_low_data, dtype='uint8')
                    del cam_low_data
                    gc.collect()

                if 'observations/images/cam_left' in infile:
                    cam_left_data = infile['observations/images/cam_left'][:]
                    outfile.create_dataset('observations/images/cam_left_wrist', data=cam_left_data, dtype='uint8')
                    del cam_left_data
                    gc.collect()


                if 'observations/images/cam_right' in infile:
                    cam_right_data = infile['observations/images/cam_right'][:]
                    outfile.create_dataset('observations/images/cam_right_wrist', data=cam_right_data, dtype='uint8')
                    del cam_right_data
                    gc.collect()


                # 创建 observations/images_depth 数据集
                depth_mapping = {
                    'observations/images_depth/cam_high': '|S58286',
                    'observations/images_depth/cam_left_wrist': '|S61818',
                    'observations/images_depth/cam_right_wrist': '|S68576'
                }

                for path, dtype in depth_mapping.items():
                    # 这里使用空字节串作为默认值
                    default_data = b''
                    target_length = int(dtype[2:])
                    padded_default_data = default_data.ljust(target_length, b'\x00')[:target_length]
                    outfile.create_dataset(path, data=[padded_default_data], dtype=dtype)

                # 复制 observations/qpos 数据集并转换为弧度制
                if 'observations/qpos' in infile:
                    qpos_data = infile['observations/qpos'][:]
                    qpos_data_rad = np.radians(qpos_data)
                    outfile.create_dataset('observations/qpos', data=qpos_data_rad, dtype='float32')
                    del qpos_data, qpos_data_rad
                    gc.collect()

                # 复制 observations/qvel 数据集
                if 'observations/qvel' in infile:
                    qvel_data = infile['observations/qvel'][:]
                    outfile.create_dataset('observations/qvel', data=qvel_data, dtype='float32')
                    del qvel_data
                    gc.collect()
                print(f"Successfully processed {input_file} after {attempt + 1} attempts.")
                return
            finally:
                # 关闭文件对象
                infile.close()
                outfile.close()
        except OSError as e:
            if attempt < max_attempts - 1:
                print(f"Error opening file {input_file} on attempt {attempt + 1}: {e}. Retrying...")
            else:
                print(f"Failed to process {input_file} after {max_attempts} attempts: {e}")


def batch_process(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.hdf5'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f'{filename}')
            print(f"Processing {input_file}...")
            process_single_file(input_file, output_file)
            print(f"Checked processing of {input_file}.")


if __name__ == "__main__":
    input_folder = './my_data'  # 替换为实际的输入文件夹路径
    output_folder = './train_data'  # 替换为实际的输出文件夹路径
    batch_process(input_folder, output_folder)