"""窗口选择与截图边界离线测试：所有图像来自合成 fixture，不调用 Win32。"""
from dataclasses import replace
import ctypes
import importlib.util
from pathlib import Path
import sys
import unittest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("window_capture", ROOT / "window_capture.py")
wc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wc
spec.loader.exec_module(wc)


def frame(size=(1280, 720)):
    image = Image.new("RGB", size, (20, 110, 65))
    ImageDraw.Draw(image).rectangle((size[0] // 4, size[1] // 3, size[0] * 3 // 4, size[1] * 2 // 3), fill=(235, 230, 210))
    return image


GAME = wc.WindowInfo(123, 456, "雀魂麻将", "Jantama_MahjongSoul.exe", 1280, 720)


class FakeBackend:
    def __init__(self, windows=None, image=None, refresh=None):
        self.windows = list(windows if windows is not None else [GAME])
        self.image = image
        self.refresh_values = iter(refresh) if refresh else None
        self.captured = []

    def list_windows(self):
        return self.windows

    def refresh_window(self, hwnd):
        if self.refresh_values is not None:
            return next(self.refresh_values)
        return next((window for window in self.windows if window.hwnd == hwnd), None)

    def capture_client(self, info):
        self.captured.append(info.hwnd)
        return self.image if self.image is not None else frame((info.client_width, info.client_height)), "fixture"


class FakeGdi:
    """可注入的 Win32 资源与绘图桩；不调用真实窗口或屏幕 API。"""
    def __init__(self, print_valid=True, screen_valid=True, responsive=True):
        self.print_valid, self.screen_valid, self.responsive = print_valid, screen_valid, responsive
        self.events = []
        self.pixels = None
        self.size = None

    def GetDC(self, hwnd):
        self.events.append(("GetDC", hwnd))
        return 1 if hwnd is not None else 4

    def ReleaseDC(self, hwnd, dc):
        self.events.append(("ReleaseDC", hwnd, dc))
        return 1

    def CreateCompatibleDC(self, dc):
        return 2

    def CreateDIBSection(self, dc, info, usage, pixels, section, offset):
        self.size = (info._obj.bmiHeader.biWidth, -info._obj.bmiHeader.biHeight)
        self.pixels = (ctypes.c_ubyte * (self.size[0] * self.size[1] * 4))()
        ctypes.cast(pixels, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.addressof(self.pixels)
        return 3

    def SelectObject(self, dc, obj):
        self.events.append(("SelectObject", obj))
        return 9 if obj == 3 else 3

    def DeleteObject(self, obj):
        self.events.append(("DeleteObject", obj))
        return 1

    def DeleteDC(self, dc):
        self.events.append(("DeleteDC", dc))
        return 1

    def SendMessageTimeoutW(self, *args):
        return self.responsive

    def _paint(self):
        raw = frame(self.size).convert("RGBA").tobytes("raw", "BGRA")
        ctypes.memmove(self.pixels, raw, len(raw))

    def PrintWindow(self, hwnd, dc, flags):
        self.events.append(("PrintWindow", flags))
        if self.print_valid:
            self._paint()
        return 1

    def GdiFlush(self):
        return 1

    def BitBlt(self, *args):
        self.events.append(("BitBlt",))
        if self.screen_valid:
            self._paint()
        return 1


def fake_win32(api, unobstructed=True):
    backend = wc.Win32Backend.__new__(wc.Win32Backend)
    backend.user = backend.gdi = api
    states = iter(unobstructed if isinstance(unobstructed, list) else [unobstructed, unobstructed])
    backend._unobstructed = lambda _: next(states)
    return backend


class WindowCaptureTests(unittest.TestCase):
    def assert_error(self, code, backend):
        with self.assertRaises(wc.WindowCaptureError) as caught:
            wc.capture_mahjong_window(backend=backend)
        self.assertEqual(caught.exception.code, code)

    def test_select_native_client_and_return_rgb_metadata(self):
        backend = FakeBackend()
        image, meta = wc.capture_mahjong_window(backend=backend)
        self.assertEqual(image.size, (1280, 720))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(meta["title"], "雀魂麻将")
        self.assertEqual(meta["clientWidth"], 1280)
        self.assertEqual(meta["method"], "fixture")
        self.assertEqual(backend.captured, [123])

    def test_browser_and_helper_are_not_game(self):
        for info in (replace(GAME, process_name="chrome.exe"), replace(GAME, process_name="msedge.exe"),
                     replace(GAME, process_name="Codex.exe"), replace(GAME, title="雀魂友人场助手")):
            backend = FakeBackend([info])
            self.assert_error("window_not_found", backend)
            self.assertEqual(backend.captured, [])

    def test_native_process_has_priority_over_title_only_match(self):
        other = replace(GAME, hwnd=234, process_name="another-game.exe")
        backend = FakeBackend([other, GAME])
        wc.capture_mahjong_window(backend=backend)
        self.assertEqual(backend.captured, [123])

    def test_unrelated_process_with_game_title_is_not_captured(self):
        backend = FakeBackend([replace(GAME, process_name="another-game.exe")])
        self.assert_error("window_not_found", backend)
        self.assertEqual(backend.captured, [])

    def test_distinct_native_windows_are_ambiguous(self):
        backend = FakeBackend([GAME, replace(GAME, hwnd=234, pid=567)])
        self.assert_error("multiple_windows", backend)
        self.assertEqual(backend.captured, [])

    def test_minimized_hidden_and_hung_windows_do_not_capture(self):
        for info, code in ((replace(GAME, minimized=True), "window_minimized"),
                           (replace(GAME, visible=False), "window_not_visible"),
                           (replace(GAME, cloaked=True), "window_not_visible"),
                           (replace(GAME, hung=True), "window_unresponsive")):
            backend = FakeBackend([info])
            self.assert_error(code, backend)
            self.assertEqual(backend.captured, [])

    def test_unknown_process_identity_is_explicit_error(self):
        self.assert_error("process_unverified", FakeBackend([replace(GAME, process_name="")]))

    def test_main_window_ignores_owned_and_tool_windows(self):
        backend = FakeBackend([replace(GAME, hwnd=234, owner=123), replace(GAME, hwnd=345, tool_window=True), GAME])
        wc.capture_mahjong_window(backend=backend)
        self.assertEqual(backend.captured, [123])

    def test_window_moved_and_scaled_needs_no_calibration(self):
        updated = replace(GAME, client_width=1920, client_height=1080, client_left=-1600, client_top=200, dpi=144)
        backend = FakeBackend([updated])
        image, meta = wc.capture_mahjong_window(backend=backend)
        self.assertEqual(image.size, (1920, 1080))
        self.assertEqual(meta["dpi"], 144)

    def test_resize_during_capture_retries(self):
        resized = replace(GAME, client_width=960, client_height=540)
        backend = FakeBackend(refresh=[GAME, resized, resized, resized])
        image, _ = wc.capture_mahjong_window(backend=backend)
        self.assertEqual(image.size, (960, 540))
        self.assertEqual(len(backend.captured), 2)

    def test_repeated_resize_does_not_return_partial_frame(self):
        resized = replace(GAME, client_width=960, client_height=540)
        again = replace(GAME, client_width=640, client_height=360)
        self.assert_error("window_resizing", FakeBackend(refresh=[GAME, resized, resized, again]))

    def test_closed_or_reused_handle_is_not_captured(self):
        self.assert_error("window_changed", FakeBackend(refresh=[None]))
        self.assert_error("window_changed", FakeBackend(refresh=[replace(GAME, pid=999)]))

    def test_black_or_flat_frame_is_explicit_error(self):
        for color in ((0, 0, 0), (255, 255, 255), (90, 90, 90)):
            self.assert_error("blank_frame", FakeBackend(image=Image.new("RGB", (1280, 720), color)))

    def test_no_windows_never_captures_desktop(self):
        backend = FakeBackend([])
        self.assert_error("window_not_found", backend)
        self.assertEqual(backend.captured, [])

    def test_invalid_client_size_is_rejected(self):
        self.assert_error("client_too_small", FakeBackend([replace(GAME, client_width=100)]))
        self.assert_error("client_too_large", FakeBackend([replace(GAME, client_width=20000, client_height=20000)]))

    def test_screen_fallback_requires_entire_client_inside_monitor(self):
        self.assertTrue(wc._screen_is_unobstructed((0, 0, 1280, 720), [], [(0, 0, 1920, 1080)]))
        self.assertFalse(wc._screen_is_unobstructed((-1, 0, 1280, 720), [], [(0, 0, 1920, 1080)]))
        self.assertFalse(wc._screen_is_unobstructed((1500, 0, 2500, 720), [], [(0, 0, 1920, 1080), (1920, 0, 3840, 1080)]))
        self.assertTrue(wc._screen_is_unobstructed((-1600, 50, -320, 770), [], [(-1920, 0, 0, 1080)]))

    def test_any_front_window_overlap_blocks_screen_fallback(self):
        target = (100, 100, 1380, 820)
        monitors = [(0, 0, 1920, 1080)]
        for occluder in ((200, 200, 300, 300), (0, 0, 2000, 2000), (1379, 819, 1400, 900)):
            self.assertFalse(wc._screen_is_unobstructed(target, [occluder], monitors))
        self.assertTrue(wc._screen_is_unobstructed(target, [(1380, 100, 1500, 820)], monitors))

    def test_printwindow_does_not_read_screen_and_releases_resources(self):
        api = FakeGdi()
        image, method = fake_win32(api).capture_client(replace(GAME, client_width=320, client_height=180))
        self.assertEqual(method, "PrintWindow")
        self.assertTrue(wc._usable_frame(image))
        self.assertNotIn(("GetDC", None), api.events)
        self.assertEqual(api.events[-4:], [("SelectObject", 9), ("DeleteObject", 3), ("DeleteDC", 2), ("ReleaseDC", 123, 1)])

    def test_black_printwindow_only_uses_guarded_client_fallback(self):
        api = FakeGdi(print_valid=False)
        image, method = fake_win32(api).capture_client(replace(GAME, client_width=320, client_height=180))
        self.assertEqual(method, "BitBlt-client-unobstructed")
        self.assertTrue(wc._usable_frame(image))
        self.assertIn(("ReleaseDC", None, 4), api.events)

    def test_occluded_window_never_reads_screen(self):
        api = FakeGdi(print_valid=False)
        with self.assertRaises(wc.WindowCaptureError) as caught:
            fake_win32(api, False).capture_client(replace(GAME, client_width=320, client_height=180))
        self.assertEqual(caught.exception.code, "window_occluded")
        self.assertNotIn(("GetDC", None), api.events)
        self.assertNotIn(("BitBlt",), api.events)
        self.assertIn(("DeleteObject", 3), api.events)

    def test_new_occlusion_after_screen_copy_discards_frame(self):
        api = FakeGdi(print_valid=False)
        with self.assertRaises(wc.WindowCaptureError) as caught:
            fake_win32(api, [True, False]).capture_client(replace(GAME, client_width=320, client_height=180))
        self.assertEqual(caught.exception.code, "window_changed")
        self.assertIn(("ReleaseDC", None, 4), api.events)

    def test_unresponsive_window_does_not_call_printwindow_or_screen(self):
        api = FakeGdi(responsive=False)
        with self.assertRaises(wc.WindowCaptureError) as caught:
            fake_win32(api).capture_client(replace(GAME, client_width=320, client_height=180))
        self.assertEqual(caught.exception.code, "window_unresponsive")
        self.assertFalse(any(event[0] in ("PrintWindow", "BitBlt") for event in api.events))
        self.assertIn(("DeleteObject", 3), api.events)

    def test_black_fallback_frame_is_never_returned(self):
        api = FakeGdi(print_valid=False, screen_valid=False)
        with self.assertRaises(wc.WindowCaptureError) as caught:
            fake_win32(api).capture_client(replace(GAME, client_width=320, client_height=180))
        self.assertEqual(caught.exception.code, "blank_frame")
        self.assertIn(("ReleaseDC", None, 4), api.events)


if __name__ == "__main__":
    unittest.main()
