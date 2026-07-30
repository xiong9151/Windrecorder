"""
WebDAV 工具模块 - 支持从 WebDAV 云盘读取视频文件

当用户将视频文件存储在支持 WebDAV 的云盘（如 Nextcloud）时，
此模块提供替代本地文件系统的视频读取能力。
"""

import os
import re
from xml.etree import ElementTree

import requests

from windrecorder.config import config
from windrecorder.logger import get_logger
from windrecorder import utils

logger = get_logger(__name__)

WEBDAV_NAMESPACE = "DAV:"

# 全局单例
_webdav_client = None


def get_webdav_client():
    """获取全局 WebDAV 客户端单例"""
    global _webdav_client
    if _webdav_client is None and config.enable_webdav_video_storage:
        if config.webdav_url and config.webdav_username:
            _webdav_client = WebDAVClient(
                base_url=config.webdav_url,
                username=config.webdav_username,
                password=config.webdav_password,
                videos_dir=config.webdav_videos_dir,
            )
    return _webdav_client


def reset_webdav_client():
    """重置客户端（配置变更时调用）"""
    global _webdav_client
    _webdav_client = None


class WebDAVClient:
    """WebDAV 客户端，封装 PROPFIND、GET 等操作"""

    def __init__(self, base_url, username, password, videos_dir="videos"):
        self.base_url = base_url.rstrip("/") + "/"
        self.videos_dir = videos_dir.strip("/")
        self.auth = (username, password)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({"User-Agent": "Windrecorder/1.0"})

    # ─── 路径构建 ───────────────────────────────────────────

    def _build_path(self, *segments):
        """拼接路径段，返回相对于 WebDAV 根目录的路径字符串（无前导斜杠）"""
        parts = [self.videos_dir]
        parts.extend(str(s).strip("/") for s in segments if s)
        parts = [p for p in parts if p]  # 过滤空字符串，避免 videos_dir 为空时产生前导斜杠
        return "/".join(parts)

    def _url(self, *segments):
        """构建完整的 WebDAV URL"""
        path = self._build_path(*segments)
        return self.base_url.rstrip("/") + "/" + path

    # ─── WebDAV 协议操作 ────────────────────────────────────

    def propfind(self, remote_path, depth="1"):
        """执行 PROPFIND 请求，返回 XML 根元素"""
        url = self.base_url + remote_path.lstrip("/")
        headers = {"Depth": depth}
        try:
            resp = self.session.request("PROPFIND", url, headers=headers, timeout=30)
            if resp.status_code in (200, 207, 301, 302, 404):
                return resp
            logger.warning(f"PROPFIND {url} returned {resp.status_code}")
            return resp
        except requests.RequestException as e:
            logger.error(f"PROPFIND {url} failed: {e}")
            return None

    def get_file(self, remote_path):
        """GET 文件内容，返回 bytes"""
        url = self.base_url + remote_path.lstrip("/")
        try:
            resp = self.session.get(url, timeout=60)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            logger.error(f"GET {url} failed: {e}")
            return None

    def get_file_stream(self, remote_path, chunk_size=8192):
        """流式读取文件，返回迭代器"""
        url = self.base_url + remote_path.lstrip("/")
        try:
            resp = self.session.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            return resp.iter_content(chunk_size=chunk_size)
        except requests.RequestException as e:
            logger.error(f"GET stream {url} failed: {e}")
            return None

    # ─── 文件/目录操作 ──────────────────────────────────────

    def list_dir(self, *segments):
        """列出目录下的所有条目（文件名），返回文件名列表"""
        remote_path = self._build_path(*segments)
        resp = self.propfind(remote_path, depth="1")
        if resp is None or resp.status_code not in (200, 207):
            return []

        try:
            root = ElementTree.fromstring(resp.content)
        except ElementTree.ParseError:
            return []

        ns = {"d": WEBDAV_NAMESPACE}
        base_href = self._extract_href(root, ns)
        items = []

        for response_elem in root.findall("d:response", ns):
            href = self._extract_href(response_elem, ns)
            if href is None or href == base_href:
                continue
            # 从 href 中提取文件名
            filename = href.rstrip("/").split("/")[-1]
            if filename:
                items.append(filename)

        return items

    def list_dir_with_href(self, *segments):
        """列出目录下的所有条目，返回 (文件名, 完整 href) 列表"""
        remote_path = self._build_path(*segments)
        resp = self.propfind(remote_path, depth="1")
        if resp is None or resp.status_code not in (200, 207):
            return []

        try:
            root = ElementTree.fromstring(resp.content)
        except ElementTree.ParseError:
            return []

        ns = {"d": WEBDAV_NAMESPACE}
        base_href = self._extract_href(root, ns)
        items = []

        for response_elem in root.findall("d:response", ns):
            href = self._extract_href(response_elem, ns)
            if href is None or href == base_href:
                continue
            filename = href.rstrip("/").split("/")[-1]
            if filename:
                items.append((filename, href))

        return items

    def file_exists(self, *segments):
        """检查文件/目录是否存在"""
        remote_path = self._build_path(*segments)
        resp = self.propfind(remote_path, depth="0")
        return resp is not None and resp.status_code in (200, 207)

    def check_video_exist(self, video_name):
        """
        检查视频文件在 WebDAV 上是否存在，返回实际文件名或 None
        优先选择未被压缩的文件
        """
        yyyy_mm = video_name[:7]
        video_prefix = video_name.split(".")[0]  # 前19个字符

        files = self.list_dir(yyyy_mm)
        if not files:
            return None

        # 匹配相同前缀的所有文件
        matches = [f for f in files if video_prefix in f]
        if not matches:
            return None

        # 优先选择未压缩的
        uncompressed = [f for f in matches if "-COMPRESS" not in f]
        if uncompressed:
            return uncompressed[0]
        return matches[0]

    def get_video_filepath(self, video_name):
        """获取视频文件的 WebDAV 相对路径"""
        yyyy_mm = video_name[:7]
        return self._build_path(yyyy_mm, video_name)

    def get_video_url(self, video_name):
        """获取视频文件的完整 WebDAV URL（含认证信息，可直接用于播放）"""
        yyyy_mm = video_name[:7]
        return self._url(yyyy_mm, video_name)

    def get_direct_playback_url(self, video_name):
        """
        获取可直接用于 HTML5 <video> 播放的 WebDAV URL。
        将认证信息嵌入 URL 中，浏览器可直接播放，不受跨域限制。
        """
        import urllib.parse

        yyyy_mm = video_name[:7]
        rel_path = self._build_path(yyyy_mm, video_name)

        parsed = urllib.parse.urlparse(self.base_url)
        if not parsed.hostname:
            return None

        # 拼接完整路径：base_url 的路径前缀 + 视频相对路径
        base_path = parsed.path.rstrip("/")
        full_path = base_path + "/" + rel_path

        # 嵌入 Basic 认证到 URL 中，确保浏览器可直接播放（解决跨域 cookie 问题）
        auth_netloc = (
            f"{urllib.parse.quote(self.auth[0], safe='')}"
            f":{urllib.parse.quote(self.auth[1], safe='')}"
            f"@{parsed.hostname}"
        )
        if parsed.port:
            auth_netloc += f":{parsed.port}"

        playback_url = parsed._replace(netloc=auth_netloc, scheme=parsed.scheme, path=full_path)
        return urllib.parse.urlunparse(playback_url)

    def get_all_video_files(self):
        """
        遍历 videos 目录，返回所有视频文件名列表。
        通过 PROPFIND 获取年月子目录，再递归获取文件。
        """
        video_files = []
        month_dirs = self.list_dir()
        for month_dir in month_dirs:
            files = self.list_dir(month_dir)
            for f in files:
                if f.endswith(".mp4"):
                    video_files.append(f)
        return video_files

    def get_video_files_by_time_range(self, start_datetime, end_datetime):
        """
        根据时间范围获取视频文件列表
        """
        all_files = self.get_all_video_files()
        result = []

        for f in all_files:
            if "-OCRED" not in f:
                continue
            try:
                file_dt = utils.dtstr_to_datetime(f[:18])
                if start_datetime <= file_dt <= end_datetime:
                    result.append(f)
            except (ValueError, IndexError):
                continue

        return result

    # ─── 内部方法 ───────────────────────────────────────────

    @staticmethod
    def _extract_href(element, ns):
        """从 PROPFIND 响应元素中提取 href"""
        href_elem = element.find("d:href", ns)
        if href_elem is not None and href_elem.text:
            return href_elem.text
        return None