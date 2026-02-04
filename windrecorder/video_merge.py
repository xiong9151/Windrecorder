import os
import subprocess
import datetime
from windrecorder.config import config
from windrecorder import utils, file_utils, db_manager, ocr_manager
from windrecorder.logger import get_logger
import pandas as pd

logger = get_logger(__name__)


def merge_short_videos_in_sequence(min_duration_minutes=14, max_duration_minutes=16):
    """
    合并连续录制的短视频，直到达到指定的时长范围
    
    :param min_duration_minutes: 最小合并时长（分钟）
    :param max_duration_minutes: 最大合并时长（分钟）
    """
    # 获取所有视频文件
    video_dirs = file_utils.get_file_dir_list_first_level(config.record_videos_dir_ud)
    video_dirs.sort(reverse=True)  # 从最新的开始处理
    
    for video_dir in video_dirs:
        video_dir_path = os.path.join(config.record_videos_dir_ud, video_dir)
        if not os.path.isdir(video_dir_path):
            continue
            
        # 获取该目录下的所有视频文件
        video_files = [f for f in os.listdir(video_dir_path) if f.endswith('.mp4') and "-OCRED" in f and "-MERGED" not in f and "-COMPRESS" not in f and "-IMGEMB" not in f]
        video_files.sort()  # 按时间顺序排序
        
        i = 0
        while i < len(video_files):
            current_video = video_files[i]
            current_video_path = os.path.join(video_dir_path, current_video)
            
            # 检查视频是否存在且可以获取信息
            if not os.path.exists(current_video_path):
                i += 1
                continue
                
            try:
                # 获取当前视频时长
                video_info = utils.get_vidfilepath_info(current_video_path)
                current_duration = float(video_info.get("duration", config.record_seconds))
            except Exception as e:
                logger.warning(f"无法获取视频信息 {current_video}: {e}")
                i += 1
                continue
            
            # 如果当前视频已经足够长，跳过
            if current_duration >= min_duration_minutes * 60:
                i += 1
                continue
                
            # 开始合并过程
            videos_to_merge = [current_video]
            total_duration = current_duration
            j = i + 1
            
            # 查找连续的视频进行合并
            while j < len(video_files) and total_duration < max_duration_minutes * 60:
                next_video = video_files[j]
                next_video_path = os.path.join(video_dir_path, next_video)
                
                # 检查是否是连续录制的视频（结束时间等于下一个视频的开始时间）
                current_end_time = utils.dtstr_to_datetime(current_video[:19]) + datetime.timedelta(seconds=total_duration)
                next_start_time = utils.dtstr_to_datetime(next_video[:19])
                
                # 允许1秒的误差
                if abs((next_start_time - current_end_time).total_seconds()) > 1:
                    break  # 不是连续的视频
                
                try:
                    next_video_info = utils.get_vidfilepath_info(next_video_path)
                    next_duration = float(next_video_info.get("duration", config.record_seconds))
                except Exception as e:
                    logger.warning(f"无法获取视频信息 {next_video}: {e}")
                    break
                    
                # 检查合并后是否会超过最大时长
                if total_duration + next_duration > max_duration_minutes * 60:
                    break
                    
                # 确保下一个视频也没有被处理过
                if "-MERGED" in next_video or "-COMPRESS" in next_video or "-IMGEMB" in next_video:
                    break
                    
                videos_to_merge.append(next_video)
                total_duration += next_duration
                j += 1
            
            # 如果有多个视频需要合并
            if len(videos_to_merge) > 1:
                merge_videos(videos_to_merge, video_dir_path, min_duration_minutes, max_duration_minutes)
                # 跳过已合并的视频
                i = j
            else:
                i += 1


def merge_videos(video_files, video_dir_path, min_duration_minutes, max_duration_minutes):
    """
    合并指定的视频文件
    
    :param video_files: 要合并的视频文件列表
    :param video_dir_path: 视频目录路径
    :param min_duration_minutes: 最小时长（分钟）
    :param max_duration_minutes: 最大时长（分钟）
    """
    if len(video_files) < 2:
        return
        
    logger.info(f"准备合并 {len(video_files)} 个视频: {video_files}")
    
    # 创建临时文件列表
    temp_list_file = os.path.join(video_dir_path, "temp_merge_list.txt")
    
    try:
        # 写入文件列表
        with open(temp_list_file, 'w', encoding='utf-8') as f:
            for video_file in video_files:
                video_path = os.path.join(video_dir_path, video_file).replace('\\', '/')
                f.write(f"file '{video_path}'\n")
        
        # 生成合并后的文件名（使用第一个视频的时间戳）
        first_video_name = video_files[0]
        # 不再添加合并后缀，直接使用第一个视频的名称（但去掉-OCRED标记）
        merged_video_name = first_video_name.replace("-OCRED", "")
        
        # 如果已存在合并后的文件，添加序号
        merged_video_path = os.path.join(video_dir_path, merged_video_name)
        counter = 1
        while os.path.exists(merged_video_path):
            name_without_ext = os.path.splitext(merged_video_name)[0]
            if "-RETRY" in name_without_ext:
                base_name = name_without_ext.split("-RETRY")[0]
                merged_video_name = f"{base_name}-RETRY{counter}{os.path.splitext(merged_video_name)[1]}"
            else:
                merged_video_name = f"{name_without_ext}-RETRY{counter}{os.path.splitext(merged_video_name)[1]}"
            merged_video_path = os.path.join(video_dir_path, merged_video_name)
            counter += 1
        
        # 使用ffmpeg合并视频
        cmd = [
            config.ffmpeg_path,
            "-f", "concat",
            "-safe", "0",
            "-i", temp_list_file,
            "-c", "copy",
            merged_video_path
        ]
        
        logger.info(f"执行合并命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"成功合并视频: {merged_video_name}")
            
            # 删除原始视频文件
            for video_file in video_files:
                original_path = os.path.join(video_dir_path, video_file)
                try:
                    os.remove(original_path)
                    logger.info(f"已删除原始视频: {video_file}")
                except Exception as e:
                    logger.error(f"删除原始视频失败 {video_file}: {e}")
            
            # 重新处理OCR等操作
            reprocess_merged_video(merged_video_name, video_dir_path, video_files)
        else:
            logger.error(f"合并视频失败: {result.stderr}")
            
    except Exception as e:
        logger.error(f"合并视频时出错: {e}")
    finally:
        # 清理临时文件
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)


def reprocess_merged_video(merged_video_name, video_dir_path, original_video_files):
    """
    重新处理合并后的视频（OCR等操作）
    
    :param merged_video_name: 合并后的视频文件名
    :param video_dir_path: 视频目录路径
    :param original_video_files: 原始视频文件列表
    """
    try:
        merged_video_path = os.path.join(video_dir_path, merged_video_name)
        
        # 检查合并后的视频是否存在
        if not os.path.exists(merged_video_path):
            logger.error(f"合并后的视频文件不存在: {merged_video_path}")
            return
            
        logger.info(f"开始处理合并后的视频: {merged_video_path}")
        
        # 为合并后的视频重新进行OCR处理
        try:
            # 获取视频所在目录和文件名
            video_dir = os.path.dirname(merged_video_path)
            video_filename = os.path.basename(merged_video_path)
            
            # 调用OCR处理函数处理合并后的视频
            ocr_manager.ocr_process_single_video(video_dir, video_filename, config.iframe_dir)
            logger.info(f"合并视频OCR处理完成: {merged_video_name}")
        except Exception as e:
            logger.error(f"合并视频OCR处理失败: {e}")
            
        # 从数据库中删除原始视频的记录
        try:
            for original_video in original_video_files:
                try:
                    db_manager.db_rollback_delete_video_refer_record(original_video)
                    logger.info(f"已从数据库删除原始视频记录: {original_video}")
                except Exception as e:
                    logger.error(f"从数据库删除原始视频记录失败 {original_video}: {e}")
        except Exception as e:
            logger.error(f"处理数据库记录时出错: {e}")
            
    except Exception as e:
        logger.error(f"重新处理合并视频时出错: {e}")


def cleanup_merged_videos():
    """
    清理处理过程中可能产生的临时文件或错误文件
    """
    pass


# 提供一个可以直接调用的函数
def process_video_merging():
    """
    主函数：处理视频合并
    """
    logger.info("开始视频合并过程")
    try:
        merge_short_videos_in_sequence()
        logger.info("视频合并过程完成")
    except Exception as e:
        logger.error(f"视频合并过程中出错: {e}")