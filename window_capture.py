"""雀魂原生窗口客户区截图；不激活窗口、不输入、不回退全桌面。

capture_mahjong_window() -> (RGB PIL.Image, metadata)
测试可传入 backend：list_windows()/refresh_window(hwnd)/capture_client(info)。
Win32 只在真正调用时加载，因此导入和离线测试不触发系统截图。
"""
from contextlib import contextmanager, nullcontext
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os
import re
from PIL import Image, ImageStat


class WindowCaptureError(RuntimeError):
    def __init__(self, code, message, *, candidates=None):
        super().__init__(message)
        self.code = code
        self.candidates = candidates or []


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    pid: int
    title: str
    process_name: str
    client_width: int
    client_height: int
    visible: bool = True
    minimized: bool = False
    cloaked: bool = False
    tool_window: bool = False
    owner: int = 0
    hung: bool = False
    dpi: int = 96
    client_left: int = 0
    client_top: int = 0


_NATIVE_PROCESSES = {"jantama_mahjongsoul.exe", "mahjongsoul.exe", "jantama.exe", "majsoul.exe"}
_EXCLUDED_PROCESSES = {
    "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "vivaldi.exe",
    "iexplore.exe", "msedgewebview2.exe", "browser.exe", "360se.exe", "360chrome.exe",
    "qqbrowser.exe", "sogouexplorer.exe", "codex.exe", "chatgpt.exe", "electron.exe",
    "code.exe", "cursor.exe", "node.exe", "python.exe", "pythonw.exe", "cmd.exe",
    "powershell.exe", "pwsh.exe", "windowsterminal.exe", "conhost.exe",
}
_HELPER_TITLE = re.compile(r"助手|辅助|mahjong[\s_-]*helper|deepseek[\s_-]*harness", re.I)
_GAME_TITLE = re.compile(r"雀魂|mahjong\s*soul|majsoul|jantama", re.I)


def _rank_window(info):
    process = info.process_name.rsplit("\\", 1)[-1].casefold()
    if process in _EXCLUDED_PROCESSES or _HELPER_TITLE.search(info.title):
        return 0
    if process in _NATIVE_PROCESSES:
        return 2
    # 只读明确的雀魂客户端进程，不能仅凭标题截取其他应用。
    return 0


def _select_window(windows):
    candidates = [w for w in windows if _rank_window(w) and not w.tool_window and not w.owner]
    if not candidates:
        if any(_GAME_TITLE.search(w.title) and not w.process_name and not _HELPER_TITLE.search(w.title) for w in windows):
            raise WindowCaptureError("process_unverified", "发现疑似雀魂窗口，但无法确认进程；请确保助手与游戏权限一致")
        raise WindowCaptureError("window_not_found", "没有找到雀魂原生游戏窗口；请先打开雀魂客户端")
    visible = [w for w in candidates if w.visible and not w.cloaked and not w.minimized]
    if not visible:
        if any(w.minimized for w in candidates):
            raise WindowCaptureError("window_minimized", "雀魂窗口已最小化；请恢复窗口后重试")
        raise WindowCaptureError("window_not_visible", "雀魂窗口当前不可见；请切回游戏所在桌面后重试")
    highest = max(_rank_window(w) for w in visible)
    visible = [w for w in visible if _rank_window(w) == highest]
    main = [w for w in visible if w.client_width >= 320 and w.client_height >= 180]
    if not main:
        raise WindowCaptureError("client_too_small", "雀魂客户区太小或尚未初始化；请稍后重试或放大窗口")
    if len(main) > 1:
        raise WindowCaptureError("multiple_windows", "发现多个雀魂窗口，请只保留一个游戏窗口后重试",
                                 candidates=[{"title": w.title, "pid": w.pid, "hwnd": w.hwnd} for w in main])
    return main[0]


def _usable_frame(image):
    if not isinstance(image, Image.Image) or image.width < 2 or image.height < 2:
        return False
    sample = image.resize((96, 64)).convert("RGB")
    stat = ImageStat.Stat(sample)
    if max(hi - lo for lo, hi in sample.getextrema()) < 8:
        return False
    raw = sample.tobytes()
    near_black = sum(1 for i in range(0, len(raw), 3) if max(raw[i:i + 3]) < 12) / (96 * 64)
    return not (near_black > 0.985 or max(stat.stddev) < 1.5)


def _screen_is_unobstructed(client_rect, front_rects, monitor_rects):
    """严格要求完整位于一个显示器内；任何上层窗口相交即拒绝屏幕拷贝。"""
    left, top, right, bottom = client_rect
    contained = any(left >= ml and top >= mt and right <= mr and bottom <= mb
                    for ml, mt, mr, mb in monitor_rects)
    if not contained:
        return False
    return not any(max(left, fl) < min(right, fr) and max(top, ft) < min(bottom, fb)
                   for fl, ft, fr, fb in front_rects)


def capture_mahjong_window(*, backend=None):
    """返回雀魂客户区图像和元数据；失败抛带 code 的 WindowCaptureError。"""
    backend = backend or Win32Backend()
    context = backend.dpi_context() if hasattr(backend, "dpi_context") else nullcontext()
    with context:
        selected = _select_window(backend.list_windows())
        for attempt in range(2):
            current = backend.refresh_window(selected.hwnd)
            if current is None or current.pid != selected.pid or not _rank_window(current):
                raise WindowCaptureError("window_changed", "雀魂窗口已关闭或发生变化，请重试")
            if current.minimized:
                raise WindowCaptureError("window_minimized", "雀魂窗口已最小化；请恢复窗口后重试")
            if not current.visible or current.cloaked:
                raise WindowCaptureError("window_not_visible", "雀魂窗口当前不可见，请恢复后重试")
            if current.hung:
                raise WindowCaptureError("window_unresponsive", "雀魂窗口暂时无响应，请等待游戏恢复")
            if current.client_width < 320 or current.client_height < 180:
                raise WindowCaptureError("client_too_small", "雀魂客户区太小，请放大窗口")
            if current.client_width * current.client_height > 40_000_000 or max(current.client_width, current.client_height) > 16384:
                raise WindowCaptureError("client_too_large", "雀魂窗口尺寸异常，请恢复普通窗口大小")
            image, method = backend.capture_client(current)
            after = backend.refresh_window(current.hwnd)
            if after is None or after.pid != selected.pid or not _rank_window(after):
                raise WindowCaptureError("window_changed", "截图时雀魂窗口已关闭，请重试")
            if after.minimized or not after.visible or after.cloaked:
                raise WindowCaptureError("window_changed", "截图时雀魂窗口可见状态改变，请重试")
            expected = (after.client_width, after.client_height)
            if expected != (current.client_width, current.client_height) or image.size != expected:
                if attempt == 0:
                    continue
                raise WindowCaptureError("window_resizing", "雀魂窗口正在缩放，请停稳后重试")
            if not _usable_frame(image):
                raise WindowCaptureError("blank_frame", "未取得有效的雀魂画面（黑屏或空白）；请等待游戏加载后重试")
            return image if image.mode == "RGB" else image.convert("RGB"), {
                "hwnd": current.hwnd, "pid": current.pid, "title": current.title,
                "processName": current.process_name, "clientWidth": after.client_width,
                "clientHeight": after.client_height, "width": after.client_width,
                "height": after.client_height, "method": method, "dpi": after.dpi,
            }
    raise WindowCaptureError("capture_unavailable", "未取得有效的雀魂客户区截图")


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class Win32Backend:
    def __init__(self):
        if os.name != "nt":
            raise WindowCaptureError("unsupported_platform", "自动窗口截图目前只支持 Windows")
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.gdi = ctypes.WinDLL("gdi32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        try:
            self.dwm = ctypes.WinDLL("dwmapi", use_last_error=True)
        except OSError:
            self.dwm = None
        self._configure_api()

    def _configure_api(self):
        u, g, k = self.user, self.gdi, self.kernel
        self.enum_callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        u.EnumWindows.argtypes = [self.enum_callback, wintypes.LPARAM]
        u.EnumWindows.restype = wintypes.BOOL
        for name in ("IsWindow", "IsWindowVisible", "IsIconic", "IsHungAppWindow"):
            getattr(u, name).argtypes = [wintypes.HWND]
            getattr(u, name).restype = wintypes.BOOL
        u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        u.GetWindowTextLengthW.restype = ctypes.c_int
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetWindowTextW.restype = ctypes.c_int
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.GetWindowThreadProcessId.restype = wintypes.DWORD
        u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.GetClientRect.restype = wintypes.BOOL
        u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        u.ClientToScreen.restype = wintypes.BOOL
        u.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
        u.GetWindow.restype = wintypes.HWND
        u.GetTopWindow.argtypes = [wintypes.HWND]
        u.GetTopWindow.restype = wintypes.HWND
        u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.GetWindowRect.restype = wintypes.BOOL
        self.monitor_callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.HDC,
                                                   ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
        u.EnumDisplayMonitors.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT), self.monitor_callback, wintypes.LPARAM]
        u.EnumDisplayMonitors.restype = wintypes.BOOL
        u.SendMessageTimeoutW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
                                         wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
        u.SendMessageTimeoutW.restype = ctypes.c_ssize_t
        self.get_style = getattr(u, "GetWindowLongPtrW", u.GetWindowLongW)
        self.get_style.argtypes = [wintypes.HWND, ctypes.c_int]
        self.get_style.restype = ctypes.c_ssize_t
        u.GetDC.argtypes = [wintypes.HWND]
        u.GetDC.restype = wintypes.HDC
        u.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        u.ReleaseDC.restype = ctypes.c_int
        u.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
        u.PrintWindow.restype = wintypes.BOOL
        if hasattr(u, "GetDpiForWindow"):
            u.GetDpiForWindow.argtypes = [wintypes.HWND]
            u.GetDpiForWindow.restype = wintypes.UINT
        if hasattr(u, "SetThreadDpiAwarenessContext"):
            u.SetThreadDpiAwarenessContext.argtypes = [wintypes.HANDLE]
            u.SetThreadDpiAwarenessContext.restype = wintypes.HANDLE
        g.CreateCompatibleDC.argtypes = [wintypes.HDC]
        g.CreateCompatibleDC.restype = wintypes.HDC
        g.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
                                      ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
        g.CreateDIBSection.restype = wintypes.HBITMAP
        g.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
        g.SelectObject.restype = wintypes.HANDLE
        g.DeleteObject.argtypes = [wintypes.HANDLE]
        g.DeleteObject.restype = wintypes.BOOL
        g.DeleteDC.argtypes = [wintypes.HDC]
        g.DeleteDC.restype = wintypes.BOOL
        g.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
        g.BitBlt.restype = wintypes.BOOL
        g.GdiFlush.argtypes = []
        g.GdiFlush.restype = wintypes.BOOL
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        k.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        if self.dwm:
            self.dwm.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
            self.dwm.DwmGetWindowAttribute.restype = ctypes.c_long

    @contextmanager
    def dpi_context(self):
        previous = None
        if hasattr(self.user, "SetThreadDpiAwarenessContext"):
            previous = self.user.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        try:
            yield
        finally:
            if previous:
                self.user.SetThreadDpiAwarenessContext(previous)

    def _process_name(self, pid):
        handle = self.kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if self.kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return os.path.basename(buffer.value)
            return ""
        finally:
            self.kernel.CloseHandle(handle)

    def refresh_window(self, hwnd):
        u = self.user
        if not u.IsWindow(hwnd):
            return None
        length = min(u.GetWindowTextLengthW(hwnd), 32767)
        title = ctypes.create_unicode_buffer(length + 1)
        u.GetWindowTextW(hwnd, title, len(title))
        pid = wintypes.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        rect = wintypes.RECT()
        if not u.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        origin = wintypes.POINT(0, 0)
        if not u.ClientToScreen(hwnd, ctypes.byref(origin)):
            return None
        cloaked = wintypes.DWORD()
        if self.dwm:
            self.dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        dpi = u.GetDpiForWindow(hwnd) if hasattr(u, "GetDpiForWindow") else 96
        return WindowInfo(int(hwnd), int(pid.value), title.value, self._process_name(pid.value),
                          rect.right - rect.left, rect.bottom - rect.top,
                          visible=bool(u.IsWindowVisible(hwnd)), minimized=bool(u.IsIconic(hwnd)),
                          cloaked=bool(cloaked.value), tool_window=bool(self.get_style(hwnd, -20) & 0x80),
                          owner=int(u.GetWindow(hwnd, 4) or 0), hung=bool(u.IsHungAppWindow(hwnd)),
                          dpi=int(dpi or 96), client_left=origin.x, client_top=origin.y)

    def list_windows(self):
        windows = []
        errors = []
        @self.enum_callback
        def collect(hwnd, _):
            try:
                item = self.refresh_window(hwnd)
                if item is not None:
                    windows.append(item)
            except Exception as error:
                errors.append(error)
            return True
        ctypes.set_last_error(0)
        if not self.user.EnumWindows(collect, 0):
            if not windows and not errors and ctypes.get_last_error() == 0:
                raise WindowCaptureError("desktop_unavailable", "当前进程无法读取桌面窗口；请在游戏所在桌面通过启动.cmd运行助手")
            raise WindowCaptureError("enumeration_failed", "无法枚举游戏窗口，请确保与游戏权限一致后重试")
        if not windows and errors:
            raise WindowCaptureError("enumeration_failed", "无法读取窗口信息，请确保与游戏权限一致")
        return windows

    def _unobstructed(self, info):
        u = self.user
        current = self.refresh_window(info.hwnd)
        if current is None or current.minimized or not current.visible or current.cloaked:
            return False
        # 屏幕法需要尺寸和位置同时稳定；PrintWindow 无此位置限制。
        geometry = lambda value: (value.pid, value.client_left, value.client_top,
                                  value.client_width, value.client_height)
        if geometry(current) != geometry(info):
            return False
        monitors = []
        @self.monitor_callback
        def monitor(_, __, rect, ___):
            value = rect.contents
            monitors.append((value.left, value.top, value.right, value.bottom))
            return True
        if not u.EnumDisplayMonitors(None, None, monitor, 0):
            return False
        front = []
        hwnd = u.GetTopWindow(None)
        seen = set()
        found = False
        while hwnd and int(hwnd) not in seen and len(seen) < 4096:
            if int(hwnd) == info.hwnd:
                found = True
                break
            seen.add(int(hwnd))
            cloaked = wintypes.DWORD()
            if self.dwm:
                self.dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
            if u.IsWindowVisible(hwnd) and not u.IsIconic(hwnd) and not cloaked.value:
                rect = wintypes.RECT()
                if not u.GetWindowRect(hwnd, ctypes.byref(rect)):
                    return False
                if self.dwm:
                    extended = wintypes.RECT()
                    if self.dwm.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(extended), ctypes.sizeof(extended)) == 0:
                        rect = extended
                if rect.right > rect.left and rect.bottom > rect.top:
                    front.append((rect.left, rect.top, rect.right, rect.bottom))
            hwnd = u.GetWindow(hwnd, 2)  # GW_HWNDNEXT，按真实 Z 顺序走到目标。
        if not found:
            return False
        client_rect = (info.client_left, info.client_top,
                       info.client_left + info.client_width, info.client_top + info.client_height)
        return _screen_is_unobstructed(client_rect, front, monitors)

    def capture_client(self, info):
        """PrintWindow 优先；仅客户区完整可见才允许屏幕区域拷贝。

        PrintWindow 是不可取消的同步 Win32 调用。750ms WM_NULL 预检用于避开
        已无响应的窗口；调用期间新发生的挂起由外层 Node worker 的30秒超时
        结束该 Python 子进程处理，不能在同一线程内可靠取消 PrintWindow。
        """
        u, g = self.user, self.gdi
        source_dc = memory_dc = bitmap = previous = None
        try:
            source_dc = u.GetDC(info.hwnd)
            if not source_dc:
                raise WindowCaptureError("capture_denied", "无法读取雀魂窗口，请确保与游戏权限一致")
            memory_dc = g.CreateCompatibleDC(source_dc)
            if not memory_dc:
                raise WindowCaptureError("capture_allocation_failed", "无法创建窗口截图资源")
            header = BITMAPINFOHEADER()
            header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            header.biWidth = info.client_width
            header.biHeight = -info.client_height
            header.biPlanes, header.biBitCount, header.biCompression = 1, 32, 0
            data = BITMAPINFO()
            data.bmiHeader = header
            pixels = ctypes.c_void_p()
            bitmap = g.CreateDIBSection(source_dc, ctypes.byref(data), 0, ctypes.byref(pixels), None, 0)
            if not bitmap or not pixels.value:
                raise WindowCaptureError("capture_allocation_failed", "无法分配雀魂截图位图")
            previous = g.SelectObject(memory_dc, bitmap)
            if not previous or previous == ctypes.c_void_p(-1).value:
                previous = None
                raise WindowCaptureError("capture_allocation_failed", "无法准备雀魂截图位图")
            size = info.client_width * info.client_height * 4
            result = ctypes.c_size_t()
            if not u.SendMessageTimeoutW(info.hwnd, 0, 0, 0, 0x22, 750, ctypes.byref(result)):
                raise WindowCaptureError("window_unresponsive", "雀魂窗口暂时无响应或权限不一致，请稍后重试")
            for flags in (3, 1):  # PW_CLIENTONLY | PW_RENDERFULLCONTENT；旧系统兼容只用 CLIENTONLY。
                ctypes.memset(pixels, 0, size)
                if not u.PrintWindow(info.hwnd, memory_dc, flags):
                    continue
                g.GdiFlush()
                image = Image.frombytes("RGB", (info.client_width, info.client_height),
                                        ctypes.string_at(pixels, size), "raw", "BGRX", 0, 1)
                if _usable_frame(image):
                    return image, "PrintWindow"
            if not self._unobstructed(info):
                raise WindowCaptureError("window_occluded", "当前渲染模式无法读取遮挡画面；请让雀魂客户区完整露出且停稳后重试")
            screen_dc = u.GetDC(None)
            if not screen_dc:
                raise WindowCaptureError("capture_denied", "无法读取雀魂可见客户区")
            try:
                ctypes.memset(pixels, 0, size)
                copied = g.BitBlt(memory_dc, 0, 0, info.client_width, info.client_height, screen_dc,
                                  info.client_left, info.client_top, 0x00CC0020 | 0x40000000)
                if not copied:
                    raise WindowCaptureError("capture_unavailable", "无法读取雀魂客户区画面")
                g.GdiFlush()
                image = Image.frombytes("RGB", (info.client_width, info.client_height),
                                        ctypes.string_at(pixels, size), "raw", "BGRX", 0, 1)
            finally:
                u.ReleaseDC(None, screen_dc)
            if not self._unobstructed(info):
                raise WindowCaptureError("window_changed", "截图时雀魂被遮挡或移动，请停稳后重试")
            if not _usable_frame(image):
                raise WindowCaptureError("blank_frame", "雀魂客户区为黑屏或空白，请等待游戏加载后重试")
            return image, "BitBlt-client-unobstructed"
        finally:
            if previous and memory_dc:
                g.SelectObject(memory_dc, previous)
            if bitmap:
                g.DeleteObject(bitmap)
            if memory_dc:
                g.DeleteDC(memory_dc)
            if source_dc:
                u.ReleaseDC(info.hwnd, source_dc)
