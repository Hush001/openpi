import h5py
import numpy as np
import cv2
import os
from pathlib import Path


def save_images(data, output_dir):
    """解码并保存JPEG图像序列"""
    os.makedirs(output_dir, exist_ok=True)
    data_bytes = data.tobytes()

    start = 0
    count = 0
    while True:
        # 查找JPEG起始标记
        start_point = data_bytes.find(b'\xff\xd8', start)
        if start_point == -1: break

        # 查找JPEG结束标记
        end_point = data_bytes.find(b'\xff\xd9', start_point)
        if end_point == -1: break
        end_point += 2  # 包含结束标记

        # 提取并解码图像
        frame_data = data_bytes[start_point:end_point]
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            filename = os.path.join(output_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(filename, img)
            count += 1

        start = end_point  # 更新搜索位置

    print(f"🖼️ 发现{count}张图像 @ {output_dir}")


def print_hdf5_structure(group: h5py.Group , prefix=''):
    print(group.name)

    for item in group:
        obj = group[item]
        if isinstance(obj, h5py.Dataset):
            print(f"{prefix}Dataset: {obj.name}")
            print(f"{prefix}  Shape: {obj.shape}")
            print(f"{prefix}  Dtype: {obj.dtype}")
            if obj.dtype == np.float32:
                print(f"{prefix}  Data0: {obj[0,:]}")

            elif obj.dtype == object:
                print(f"{prefix}  Alldt: {obj[()]}")

            else:
                print(f"{prefix}  Alldt: {obj}") #obj is jpeg


        elif isinstance(obj, h5py.Group):
            print_hdf5_structure(obj, prefix + '  ')


def save_hdf5_structure(group: h5py.Group ,base_path:Path ,prefix=''):

    for item in group:
        obj = group[item]
        if isinstance(obj, h5py.Dataset):
            if obj.dtype == np.float32:
                full_path = obj.name
                output_path = os.path.join(base_path, full_path.lstrip('/'))
                output_path = output_path + '.csv'
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.savetxt(output_path, obj[:,:], delimiter=',')

                print(output_path)

            elif obj.dtype == object:
                full_path = obj.name
                output_path = os.path.join(base_path, full_path.lstrip('/'))
                output_path = output_path + '.txt'
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as file:
                    file.write(str(obj[()]))
                print(output_path)

            else:
                full_path = obj.name
                output_path = os.path.join(base_path, full_path.lstrip('/'))
                save_images(obj[()], output_path)

        elif isinstance(obj, h5py.Group):
            save_hdf5_structure(obj)


if __name__ == "__main__":
    with h5py.File("./raw_data/episode_0.hdf5", 'r') as f:
        print_hdf5_structure(f)
        #save_hdf5_structure(f,"../lerobot_data")
