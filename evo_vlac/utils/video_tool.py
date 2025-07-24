from loguru import logger
import traceback
import numpy as np
import torch
import json
import os
import io
import shutil
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import textwrap
import random
import copy
import re
import tempfile
import cv2
import subprocess


def images_get_from_video(video_path):
    """
    将 MP4 视频转换为 PIL Image 对象列表
    
    参数:
        video_path (str): 视频文件的路径
        
    返回:
        list: 包含视频每一帧的 PIL Image 对象的列表
    """
    import cv2
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not video.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    image_list = []
    
    # 逐帧读取视频
    while True:
        # 读取一帧
        ret, frame = video.read()
        
        # 如果读取失败，说明到达视频末尾
        if not ret:
            break
        
        # 将 OpenCV 的 BGR 格式转换为 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 将 NumPy 数组转换为 PIL Image 对象
        pil_image = Image.fromarray(rgb_frame)
        
        # 将图像添加到列表
        image_list.append(pil_image)
    
    # 释放视频资源
    video.release()
    
    return image_list

def video_trajectory(traj_id, view, image_paths, image_objects, done_list, critic_list, value_list, task, output_path, fps=5, ref_image_paths=None, n_num=5, critic_log_scale=False):
    """
    将轨迹图片与done和value曲线构建为视频，并在上方添加参考图片
    
    参数:
    - traj_id: 轨迹ID
    - view: 视角
    - image_paths: 图片路径列表
    - image_objects: 图片对象列表
    - done_list: done状态列表
    - critic_list: critic评价列表
    - value_list: 价值评估列表
    - task: 任务描述
    - output_path: 输出路径
    - fps: 视频帧率
    - ref_image_paths: 参考图片路径列表(也可以是对象)
    - n_num: 要展示的参考图片数量
    - critic_log_scale: 是否对critic使用对数坐标
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    import matplotlib
    import subprocess
    import tempfile
    import shutil
    matplotlib.use('Agg')  # 非交互式后端，避免显示图形
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 视频文件路径
    video_path = os.path.join(output_path, f"{traj_id}_{view}_trajectory.mp4")
    
    # 计算skip值 - 关键修改点
    skip = 1
    if value_list and len(image_objects) > len(value_list):
        skip = round(len(image_objects) / len(value_list))
        print(f"计算得到的skip值: {skip}")
    
    # 转换done_list为数值（如果不为None）
    done_values = None
    if done_list is not None and len(done_list) > 0:
        try:
            done_values = [float(d) if d is not None else 0 for d in done_list]
        except:
            done_values = None
    
    # 转换critic_list为数值（如果不为None）
    critic_values = None
    if critic_list is not None and len(critic_list) > 0:
        try:
            critic_values = [float(d) if d is not None else 0 for d in critic_list]
        except:
            critic_values = None
    
    # 转换value_list为数值（如果不为None）
    value_values = None
    if value_list is not None and len(value_list) > 0:
        try:
            value_values = [float(d) if d is not None else 0 for d in value_list]
        except:
            value_values = None
    
    # 处理critic_list，使其前方补0与value_list长度一致
    if critic_values and value_values and len(critic_values) < len(value_values):
        padding_length = len(value_values) - len(critic_values)
        critic_values = [0] * padding_length + critic_values
    
    if len(image_paths) > len(image_objects):
        image_objects = [Image.open(img) for img in image_paths]
    
    resized_images = []

    for img in image_objects:
        # 获取原始图片的宽度和高度
        img_width, img_height = img.size
        
        # 计算缩放比例
        scale_factor = 480 / img_height
        
        # 计算新的宽度和高度
        new_width = int(img_width * scale_factor)
        new_height = 480  # 高度固定为 480
        
        # 调整图片大小，使用新的 Resampling 方法
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 保存调整后的图片到新列表
        resized_images.append(resized_img)
    image_objects=resized_images
    # 获取图片尺寸
    sample_img = image_objects[0]
    img_width, img_height = sample_img.size
    
    # 处理参考图片
    ref_images = []
    if ref_image_paths and n_num > 0:
        # 等间隔采样参考图片
        if len(ref_image_paths) <= n_num:
            indices = range(len(ref_image_paths))
        else:
            indices = np.linspace(0, len(ref_image_paths) - 1, n_num, dtype=int)
        
        for idx in indices:
            try:
                if type(ref_image_paths[idx]) is str:
                    ref_img = Image.open(ref_image_paths[idx])
                else:
                    ref_img = ref_image_paths[idx]
                ref_images.append(ref_img)
            except Exception as e:
                print(f"无法加载参考图片 {ref_image_paths[idx]}: {e}")
    
    # 计算参考图片区域的高度和布局
    ref_height = 0
    reference_text_width = 80
    
    # 计算需要绘制的曲线数量
    curves_count = 0
    if done_values is not None:
        curves_count += 1
    if value_values is not None:
        curves_count += 1
    if critic_values is not None:
        curves_count += 1
    
    # 根据曲线数量调整布局
    if curves_count == 0:
        plot_width = 200  # 如果没有曲线，只留信息区域
    else:
        plot_width = 448  # 有曲线时的宽度
    
    # 设置视频尺寸 - 确保所有内容都能显示
    frame_width = img_width + plot_width
    frame_height = img_height + (ref_height + 30 if ref_images else 0)  # 30px间距

    if ref_images:
        # 参考图片高度为主图高度的1/3，提升显示效果
        ref_height = min(img_height // 3, 150)  # 增加最大高度
        
        # 计算单个参考图片的宽度
        single_ref_width = int(ref_height * (ref_images[0].width / ref_images[0].height))
        
        # 确保参考图片序列不会超出整个frame宽度(包括右侧区域)
        total_available_width = frame_width - reference_text_width - 40  # 留40px边距
        max_ref_width = total_available_width // len(ref_images)
        
        if single_ref_width > max_ref_width:
            single_ref_width = max_ref_width
            ref_height = int(single_ref_width * (ref_images[0].height / ref_images[0].width))
    
    # 创建临时目录保存帧图片
    temp_dir = tempfile.mkdtemp()
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 14)  # 稍微缩小字体
        small_font = ImageFont.truetype("arial.ttf", 12)
        ref_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        ref_font = ImageFont.load_default()
    
    try:
        # 对每一帧图片生成视频帧
        for i in range(len(image_objects)):
            # 创建一个空白画布
            frame = Image.new('RGB', (frame_width, frame_height), color='white')
            
            # 如果有参考图片，先添加参考图片和Reference标签
            if ref_images:
                # 添加"Reference"文字标签
                draw = ImageDraw.Draw(frame)
                ref_text_y = (ref_height - 20) // 2
                draw.text((10, ref_text_y), "Reference", fill="black", font=ref_font)
                
                # 绘制参考图片
                start_x = reference_text_width
                for j, ref_img in enumerate(ref_images):
                    # 调整参考图片大小
                    resized_ref = ref_img.resize((single_ref_width, ref_height), Image.LANCZOS)
                    # 可以跨越到右侧区域
                    paste_x = start_x + j * single_ref_width
                    frame.paste(resized_ref, (paste_x, 0))
            
            # 粘贴当前图片
            main_img_y = ref_height + 20 if ref_images else 0
            frame.paste(image_objects[i], (0, main_img_y))
            
            # 创建绘图对象
            draw = ImageDraw.Draw(frame)
            
            # 计算信息显示区域
            info_x = img_width + 10
            info_y = main_img_y + 10
            
            # 添加任务描述
            task_lines = textwrap.wrap(f"Task: {task}", width=25)  # 缩短任务描述宽度
            task_width = 0
            for line_idx, line in enumerate(task_lines[:4]):  # 最多显示2行
                draw.text((info_x-5, info_y + line_idx * 18), line, fill="black", font=font)
                # 计算任务文本的实际宽度
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                task_width = max(task_width, line_width)
            
            # 在任务描述右侧横向排列状态信息
            status_x = info_x + task_width + 30  # 任务文本右侧30px处开始
            status_y = info_y
            
            # 计算当前帧在value_list中的对应索引 - 关键修改点
            value_idx = i // skip
            
            # 横向排列当前帧状态信息
            current_x = status_x
            if done_list and value_idx < len(done_list):
                done_status = done_list[value_idx]
                done_color = "green" if done_status == "Yes" or done_status == True or done_status == 1 else "red"
                done_text = f"Done: {done_status}"
                draw.text((current_x, status_y), done_text, fill=done_color, font=font)
                # 计算文本宽度并移动到下一个位置
                bbox = draw.textbbox((0, 0), done_text, font=font)
                current_x += (bbox[2] - bbox[0]) + 20
            
            if value_list and value_idx < len(value_list):
                value = value_list[value_idx]
                value_color = "green" if value > 0 else "red"
                value_text = f"Value: {value:.4f}"
                draw.text((current_x, status_y), value_text, fill=value_color, font=font)
                # 计算文本宽度并移动到下一个位置
                bbox = draw.textbbox((0, 0), value_text, font=font)
                current_x += (bbox[2] - bbox[0]) + 20
            
            if critic_list and value_idx < len(critic_list):
                critic = float(critic_list[value_idx])
                critic_color = "green" if critic >= 0 else "red"
                critic_text = f"Critic: {critic:.4f}"
                draw.text((current_x, status_y), critic_text, fill=critic_color, font=font)
            
            # # 在第二行显示帧计数和value索引
            # draw.text((status_x, status_y + 20), f"Frame: {i+1}/{len(image_objects)} (Value idx: {value_idx})", fill="black", font=small_font)
            
            # 如果有数据，创建曲线图 - 关键修改点
            if curves_count > 0 and value_values:
                # 固定字体大小
                plt.rcParams.update({
                    'font.size': 10,
                    'axes.titlesize': 11,
                    'axes.labelsize': 10,
                    'xtick.labelsize': 8,
                    'ytick.labelsize': 8
                })
                # 计算适当的高度，保持图表的可读性
                plot_height = frame_height - status_y - 70  # 为上方信息留出更多空间
                
                # 根据可用空间确定图形尺寸
                available_width = plot_width - 30  # 留出一些边距
                fig_width = available_width / 100  # 转换为英寸
                fig_height = plot_height/100
                
                fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
                
                # 调整子图布局
                plt.subplots_adjust(hspace=0.4, left=0.15, right=0.95, bottom=0.12, top=0.9)
                
                subplot_idx = 1
                
                # 计算当前应该显示到哪个value点
                current_value_idx = min(value_idx, len(value_values) - 1)

                
                # 绘制value曲线
                if value_values is not None:
                    ax = fig.add_subplot(curves_count, 1, subplot_idx)
                    if current_value_idx >= 0:
                        # 绘制value曲线，根据critic状态着色
                        for j in range(current_value_idx):
                            color = 'orange' if (critic_values and j < len(critic_values) and critic_values[j] < 0) else 'green'
                            ax.plot([j, j+1], [value_values[j], value_values[j+1]], color=color, alpha=0.7)
                            marker_color = 'orange' if (critic_values and j < len(critic_values) and critic_values[j] < 0) else 'green'
                            ax.plot(j, value_values[j], marker='o', color=marker_color, markersize=2)
                        
                        # 当前点
                        current_color = 'orange' if (critic_values and current_value_idx < len(critic_values) and critic_values[current_value_idx] < 0) else 'green'
                        ax.plot(current_value_idx, value_values[current_value_idx], marker='o', color=current_color, markersize=4)
                    
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                    # 只在最后一个子图显示x轴标签
                    if subplot_idx == curves_count:
                        ax.set_xlabel('Step')
                    else:
                        ax.set_xticklabels([])  # 隐藏中间子图的x轴标签
                    subplot_idx += 1
                
                # 绘制critic曲线
                if critic_values is not None:
                    ax = fig.add_subplot(curves_count, 1, subplot_idx)
                    if current_value_idx >= 0:
                        critic_data = np.array(critic_values[:current_value_idx+1])
                        
                        # 绘制线条
                        ax.plot(range(current_value_idx+1), critic_data, 'r-', alpha=0.7)
                        
                        # 绘制点
                        for j in range(current_value_idx+1):
                            color = 'orange' if critic_values[j] < 0 else 'red'
                            ax.plot(j, critic_data[j], marker='o', color=color, markersize=2)
                        
                        # 当前点
                        current_color = 'orange' if critic_values[current_value_idx] < 0 else 'red'
                        ax.plot(current_value_idx, critic_data[current_value_idx], marker='o', color=current_color, markersize=4, 
                            markeredgecolor='black', markeredgewidth=0.5)
                        
                        # 设置坐标轴
                        if critic_log_scale and len(critic_data) > 0:
                            ax.set_yscale('symlog')
                            ax.set_ylabel('Critic (log)')
                        else:
                            ax.set_ylabel('Critic')
                    
                    ax.grid(True, alpha=0.3)
                    # 只在最后一个子图显示x轴标签
                    if subplot_idx == curves_count:
                        ax.set_xlabel('Step')
                    else:
                        ax.set_xticklabels([])  # 隐藏中间子图的x轴标签
                    subplot_idx += 1
                
                # 绘制done曲线
                if done_values is not None:
                    ax = fig.add_subplot(curves_count, 1, subplot_idx)
                    if current_value_idx >= 0:
                        ax.plot(done_values[:current_value_idx+1], 'b-', marker='o', markersize=2, label='Done')
                        ax.plot([current_value_idx], [done_values[current_value_idx]], 'ro', markersize=4)
                    ax.set_ylim(-0.1, 1.1)
                    ax.set_ylabel('Done')
                    ax.grid(True, alpha=0.3)
                    # 只在最后一个子图显示x轴标签
                    if subplot_idx == curves_count:
                        ax.set_xlabel('Step')
                    else:
                        ax.set_xticklabels([])  # 隐藏中间子图的x轴标签
                    subplot_idx += 1
                
                # 将matplotlib图转换为PIL图像
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                w, h = fig.canvas.get_width_height()
                plot_img = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1).convert('RGB')
                plt.close(fig)
                
                # 调整曲线图大小并粘贴到帧上
                plot_img = plot_img.resize((available_width, plot_height), Image.LANCZOS)
                paste_y = status_y + 50  # 在状态信息下方50px处开始绘制曲线
                frame.paste(plot_img, (info_x, paste_y))
            
            # 保存帧为PNG文件
            frame_path = os.path.join(temp_dir, f"{i:08d}.png")
            frame.save(frame_path, "PNG")
        
        # 使用FFmpeg生成视频（更好的兼容性）
        try:
            # 检查FFmpeg是否可用
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # 使用FFmpeg生成视频
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # 覆盖输出文件
                "-framerate", str(fps),
                "-i", os.path.join(temp_dir, "%08d.png"),
                "-c:v", "libx264",  # 使用H.264编码
                "-profile:v", "high",
                "-pix_fmt", "yuv420p",  # 确保兼容性
                "-crf", "23",  # 控制质量（0-51，数值越小质量越高）
                "-movflags", "+faststart",  # 优化网络播放
                video_path
            ]
            
            print("正在使用FFmpeg生成视频...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"视频已保存到 {video_path}")
            else:
                print(f"FFmpeg错误: {result.stderr}")
                # 如果FFmpeg失败，回退到OpenCV方法
                raise subprocess.CalledProcessError(result.returncode, ffmpeg_cmd)
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print("FFmpeg不可用或失败，回退到OpenCV方法...")
            
            # 回退到OpenCV方法
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
            
            for i in range(len(image_objects)):
                frame_path = os.path.join(temp_dir, f"{i:08d}.png")
                frame_img = cv2.imread(frame_path)
                if frame_img is not None:
                    video_writer.write(frame_img)
            
            video_writer.release()
            print(f"视频已保存到 {video_path}")
    
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return video_path


def visualize_le(image_objects, critic_list, value_list, task, id, output_path, resize_shape=(640, 480), fps=5,
                 transparency=0.1, line_width=4, edge_width=2):
    #### 只画value_list
    import matplotlib
    matplotlib.use('Agg')
    import subprocess
    temp_dir = tempfile.mkdtemp()
    os.makedirs(output_path, exist_ok=True)
    video_path = os.path.join(output_path, f"{task}_{id}_trajectory_legend.mp4")
    skip = 1
    skip_step = 0
    if value_list and len(image_objects) > len(value_list):
        skip = round(len(image_objects) / len(value_list))
        print(f"计算得到的skip值: {skip}")
        ### 计算跳帧数
        skip_step = len(image_objects) - len(value_list)
    if value_list is not None and len(value_list) > 0:
        try:
            value_values = [float(d) if d is not None else 0 for d in value_list]
        except:
            value_values = None
    critic_values = None
    if critic_list is not None and len(critic_list) > 0:
        try:
            critic_values = [float(d) if d is not None else 0 for d in critic_list]
        except:
            critic_values = None

    ### 图片resize
    ### 暂不考虑ref_images
    frame_width, frame_height = resize_shape
    resized_images = []
    for img in image_objects:
        resized_images.append(img.resize(resize_shape, Image.Resampling.LANCZOS))

    ### 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 14)  # 稍微缩小字体
        small_font = ImageFont.truetype("arial.ttf", 12)
        ref_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        ref_font = ImageFont.load_default()

    try:
        # 对每一帧图片生成视频帧
        for i in range(len(image_objects)):
            # 创建一个空白画布并粘贴画面
            frame = Image.new('RGB', (frame_width, frame_height), color='white')
            frame.paste(resized_images[i], (0, 0))
            draw = ImageDraw.Draw(frame)

            # 计算当前帧在value_list中的对应索引 - 关键修改点
            value_idx = i // skip - skip_step

            # 如果有数据，创建曲线图 - 关键修改点
            if value_idx >= 0:
                fig = plt.figure(figsize=(frame_width / 100, frame_height / 100), dpi=100)
                fig.patch.set_alpha(transparency)
                ax = fig.add_axes([0, 0, 1, 1])
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                ax.axis('off')
                plt.xlim(0, len(value_values)-1)
                plt.ylim(min(value_values) - abs(max(value_values)) * 0.15, max(value_values) + abs(max(value_values)) * 0.15)
                current_value_idx = min(value_idx, len(value_values) - 1)
                if current_value_idx >= 0:
                    for j in range(current_value_idx):
                        ax.plot([j, j + 1], [value_values[j], value_values[j + 1]], color='white', alpha=0.3, linewidth=line_width+2*edge_width)
                        color = '#4B0082' if (
                                critic_values and j < len(critic_values) and critic_values[j] < 0) else '#FF4500'
                        ax.plot([j, j + 1], [value_values[j], value_values[j + 1]], color=color, alpha=0.5, linewidth=line_width)
                        marker_color = '#4B0082' if (
                                critic_values and j < len(critic_values) and critic_values[j] < 0) else '#FF4500'
                        ax.plot(j, value_values[j], marker='o', color='black', markersize=line_width + 2, alpha=0.5)
                        ax.plot(j, value_values[j], marker='o', color=marker_color, markersize=line_width, alpha=0.5)


                    # 将matplotlib图转换为PIL图像
                    fig.canvas.draw()
                    buf = fig.canvas.buffer_rgba()
                    w, h = fig.canvas.get_width_height()
                    plot_img = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1)# .convert('RGB')
                    plt.close(fig)
                    # frame.paste(plot_img, (0, 0), mask=plot_img.convert("RGBA"))
                    frame = Image.alpha_composite(frame.convert('RGBA'), plot_img).convert('RGB')

            # 保存帧为PNG文件
            frame_path = os.path.join(temp_dir, f"{i:08d}.png")
            frame.save(frame_path, "PNG")

        # 使用FFmpeg生成视频（更好的兼容性）
        try:
            # 检查FFmpeg是否可用
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            # 使用FFmpeg生成视频
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # 覆盖输出文件
                "-framerate", str(fps),
                "-i", os.path.join(temp_dir, "%08d.png"),
                "-c:v", "libx264",  # 使用H.264编码
                "-profile:v", "high",
                "-pix_fmt", "yuv420p",  # 确保兼容性
                "-crf", "23",  # 控制质量（0-51，数值越小质量越高）
                "-movflags", "+faststart",  # 优化网络播放
                video_path
            ]

            print("正在使用FFmpeg生成视频...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"视频已保存到 {video_path}")
            else:
                print(f"FFmpeg错误: {result.stderr}")
                # 如果FFmpeg失败，回退到OpenCV方法
                raise subprocess.CalledProcessError(result.returncode, ffmpeg_cmd)

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print("FFmpeg不可用或失败，回退到OpenCV方法...")

            # 回退到OpenCV方法
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

            for i in range(len(image_objects)):
                frame_path = os.path.join(temp_dir, f"{i:08d}.png")
                frame_img = cv2.imread(frame_path)
                if frame_img is not None:
                    video_writer.write(frame_img)

            video_writer.release()
            print(f"视频已保存到 {video_path}")

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

    return video_path


def read_data_and_create_video(filename, camera_name=None, create_video=False, output_path=None, fps=30, start_frame=0, end_frame=None):
    """
    读取存储的压缩pickle文件，并可选地将其中的图片生成视频
    
    参数:
        filename (str): 数据文件路径
        camera_name (str, optional): 要处理的相机名称，如果为None则处理第一个找到的相机
        create_video (bool, optional): 是否创建视频，默认为False
        output_path (str, optional): 视频输出路径，如果为None且create_video=True，则使用与输入文件相同的名称
        fps (int, optional): 视频帧率，默认为30
        start_frame (int, optional): 起始帧索引，默认为0
        end_frame (int, optional): 结束帧索引，默认为None表示处理到最后一帧
        
    返回:
        list: PIL.Image对象列表
    """
    import gzip
    import pickle
    import base64
    import numpy as np
    import os
    import cv2
    import tempfile
    import subprocess
    from PIL import Image
    import io
    import shutil
    
    # 读取压缩的pickle文件
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # 确定处理的帧范围
    if end_frame is None:
        end_frame = len(data)
    else:
        end_frame = min(end_frame, len(data))
    
    # 确保起始帧有效
    start_frame = max(0, min(start_frame, len(data) - 1))
    
    # 如果未指定相机名称，使用第一个可用的相机
    print(data[0]['rgb'].keys())
    if camera_name is None and len(data) > 0:
        camera_name = list(data[0]['rgb'].keys())[0]
    
    # 存储解码后的图像
    images = []
    
    # 创建临时目录用于存储视频帧
    temp_dir = None
    if create_video:
        temp_dir = tempfile.mkdtemp()
    
    try:
        # 处理每一帧
        for i, frame_data in enumerate(data[start_frame:end_frame]):
            if camera_name not in frame_data['rgb']:
                print(f"警告: 帧 {i+start_frame} 中没有找到相机 {camera_name}")
                continue
            
            # 解码图像
            img_base64 = frame_data['rgb'][camera_name]
            img_data = base64.b64decode(img_base64)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # BGR转RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            images.append(pil_img)
            
            # 如果需要创建视频，保存帧到临时目录
            if create_video:
                frame_path = os.path.join(temp_dir, f"{i+start_frame:08d}.png")
                pil_img.save(frame_path)
        
        # 创建视频
        if create_video and images:
            if output_path is None:
                # 如果未指定输出路径，创建与输入文件同名的文件夹
                base_name = os.path.splitext(os.path.basename(filename))[0]
                output_dir = os.path.join(os.path.dirname(filename), base_name)
                
                # 确保输出目录存在
                os.makedirs(output_dir, exist_ok=True)
                
                # 在该目录中创建视频文件
                output_path = os.path.join(output_dir, f"{camera_name}_video.mp4")
            
            # 检查ffmpeg是否可用
            try:
                subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                # 使用FFmpeg生成视频
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",  # 覆盖输出文件
                    "-framerate", str(fps),
                    "-i", os.path.join(temp_dir, "%08d.png"),
                    "-c:v", "libx264",  # 使用H.264编码
                    "-profile:v", "high",
                    "-pix_fmt", "yuv420p",  # 确保兼容性
                    "-crf", "23",  # 控制质量（0-51，数值越小质量越高）
                    "-movflags", "+faststart",  # 优化网络播放
                    output_path
                ]
                
                subprocess.run(ffmpeg_cmd, check=True)
                print(f"视频已保存至: {output_path}")
            except subprocess.CalledProcessError:
                print("错误: 无法运行FFmpeg，请确保它已正确安装")
            except Exception as e:
                print(f"创建视频时出错: {str(e)}")
    
    finally:
        # 清理临时目录
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return images

# 视频压缩函数
def compress_video(input_path, output_path, target_size=(448, 448), fps=5):
    """
    压缩视频到指定大小和帧率
    如果原视频fps小于指定值则保持原帧率
    """
    cap = cv2.VideoCapture(input_path)
    
    # 获取原始视频的帧率和尺寸
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 如果原始帧率小于指定帧率，则使用原始帧率
    output_fps = original_fps if original_fps < fps else fps
    if target_size is None:
        target_size = (original_width, original_height)
    
    sampling_interval = int(original_fps / output_fps) if output_fps < original_fps else 1
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, target_size)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # # 处理宽高比
        # if original_width > original_height * 1.5:
        #     # 宽大于高的1.5倍，需要中心裁剪
        #     target_width = int(original_height * 1.5)
        #     # 计算裁剪的起始x坐标（居中裁剪）
        #     start_x = int((original_width - target_width) / 2)
        #     # 裁剪frame
        #     frame = frame[:, start_x:start_x+target_width]
        
        # 只有当前帧是采样帧时才处理和写入
        if frame_count % sampling_interval == 0:
            # 缩放到目标尺寸
            resized = cv2.resize(frame, target_size)
            out.write(resized)
        frame_count += 1
    
    cap.release()
    out.release()
    return output_path,output_fps

# 图片压缩函数
def compress_image(input_path, output_path, target_size=(448, 448)):
    """压缩图片到指定大小"""
    img = Image.open(input_path)
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img.save(output_path)
    return output_path
