"""Desktop resources, user-scoped encrypted configuration, and exact JS calculations."""
from collections import Counter
from dataclasses import asdict, dataclass
import base64
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import sys
import threading
from datetime import datetime

ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
_INSTANCE_MUTEX = None
LOG_NAME = 'diagnostics.log'
if (ROOT / '.python-deps').is_dir():
    sys.path.insert(0, str(ROOT / '.python-deps'))

APP_NAME = 'MahjongStrategyAssistant'
TILE_RE = re.compile(r'^(?:[0-9][mps]|[1-7]z)$')


def user_directory():
    path = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local')) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_event(message, level='INFO'):
    """Append a credential-free diagnostic line. Never pass keys or raw output."""
    text = str(message)[:400]
    text = re.sub(r'Bearer\s+\S+', 'Bearer [redacted]', text, flags=re.IGNORECASE)
    text = re.sub(r'(?i)(api[_-]?key\s*[=:]\s*)\S+', r'\1[redacted]', text)
    try:
        target = user_directory() / LOG_NAME
        stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with target.open('a', encoding='utf-8') as stream:
            stream.write('%s [%s] %s\n' % (stamp, level, text))
    except Exception:
        pass


def acquire_single_instance(name='Local\\MahjongStrategyAssistant_Instance'):
    """Windows named mutex so a second process cannot bypass the in-app lock."""
    global _INSTANCE_MUTEX
    if os.name != 'nt':
        return True
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    ctypes.set_last_error(0)
    handle = kernel.CreateMutexW(None, True, name)
    if not handle:
        raise OSError('无法创建单实例锁')
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        kernel.CloseHandle(handle)
        return False
    _INSTANCE_MUTEX = handle  # keep reference alive for process lifetime
    return True


class _Blob(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]


def _crypt(value, decrypt=False):
    if os.name != 'nt':
        raise RuntimeError('密钥加密目前只支持 Windows')
    data = (ctypes.c_ubyte * max(1, len(value)))()
    if value:
        ctypes.memmove(data, value, len(value))
    source = _Blob(len(value), ctypes.cast(data, ctypes.POINTER(ctypes.c_ubyte)))
    target = _Blob()
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
    function.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise OSError('无法读取当前 Windows 用户的加密密钥')
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel.LocalFree(target.data)


@dataclass
class Settings:
    base_url: str = 'http://10.201.137.10:8080/v1'
    model: str = 'qwen3.8-flash-next-iq3_s'
    api_key: str = ''
    reasoning_effort: str = 'medium'
    max_tokens: int = 16384
    timeout: int = 600
    capture_source: str = 'window'
    analysis_mode: str = 'structured'

    def validate(self):
        from urllib.parse import urlsplit
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('模型地址应为有效的 http/https Base URL，不能包含用户名或密码')
        if not self.model.strip():
            raise ValueError('请填写模型名称')
        if self.reasoning_effort not in ('low', 'medium', 'high'):
            raise ValueError('策略分析必须启用思考（low / medium / high）')
        if not isinstance(self.max_tokens, int) or not 2048 <= self.max_tokens <= 32768:
            raise ValueError('输出上限应在 2048 到 32768 之间')
        if not isinstance(self.timeout, int) or not 60 <= self.timeout <= 1800:
            raise ValueError('等待时间应在 60 到 1800 秒之间')
        if self.capture_source not in ('window', 'screen'):
            raise ValueError('截图来源无效')
        if self.analysis_mode not in ('fast', 'structured', 'full'):
            raise ValueError('分析模式无效')
        return self

    @property
    def endpoint(self):
        value = self.base_url.rstrip('/')
        return value if value.endswith('/chat/completions') else value + '/chat/completions'


def save_settings(settings, path=None):
    settings.validate()
    target = Path(path) if path else user_directory() / 'settings.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    value = asdict(settings)
    key = value.pop('api_key')
    value['api_key_dpapi'] = base64.b64encode(_crypt(key.encode('utf-8'))).decode('ascii') if key else ''
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, target)


def load_settings(path=None):
    target = Path(path) if path else user_directory() / 'settings.json'
    if not target.is_file():
        return Settings()
    value = json.loads(target.read_text(encoding='utf-8-sig'))
    encrypted = value.pop('api_key_dpapi', '')
    allowed = Settings.__dataclass_fields__
    result = Settings(**{k: v for k, v in value.items() if k in allowed and k != 'api_key'})
    result.api_key = _crypt(base64.b64decode(encrypted), decrypt=True).decode('utf-8') if encrypted else ''
    return result.validate()


def tile_index(code):
    if not isinstance(code, str) or not TILE_RE.fullmatch(code):
        raise ValueError('牌名无效')
    number = 5 if code[0] == '0' else int(code[0])
    return {'m': 0, 'p': 9, 's': 18, 'z': 27}[code[1]] + number - 1


def tile_name(code):
    if not isinstance(code, str) or not TILE_RE.fullmatch(code):
        return str(code)
    if code[1] == 'z':
        return ['东', '南', '西', '北', '白', '发', '中'][int(code[0]) - 1]
    return ('赤' if code[0] == '0' else '') + ('5' if code[0] == '0' else code[0]) + {'m': '万', 'p': '筒', 's': '索'}[code[1]]


def dora_from_indicator(code):
    tile_index(code)
    number = 5 if code[0]=='0' else int(code[0])
    if code[1]!='z':
        return str(number%9+1)+code[1]
    # Single honor cycle 東南西北白發中→東: the successor of 北 is 白 and of
    # 中 is 東, per standard riichi dora rules — not two separate cycles.
    return str(number%7+1)+'z'


def build_seen(hand_codes, evidence, min_score=0.65):
    """Derive visible-tile counts from local candidates for the exact solver.

    Only high-confidence candidates are hard-counted; the JS core rejects any
    tile whose hand+seen total exceeds four, so every count is capped by the
    number of physical copies not already held. Returns the code list and an
    audit dict that must be surfaced to the user, since river/meld candidates
    are unverified and low-confidence piles are deliberately excluded.
    """
    hand_counts = Counter(tile_index(code) for code in hand_codes if TILE_RE.fullmatch(str(code)))
    candidates = []
    for seat, items in (evidence.get('rivers') or {}).items():
        for item in items or []:
            if isinstance(item, dict) and item.get('name'):
                candidates.append((item['name'], item.get('score'), '牌河'))
    for seat, meld in (evidence.get('melds') or {}).items():
        for group in (meld or {}).get('groups', []) or []:
            for item in group or []:
                if isinstance(item, dict) and item.get('name'):
                    candidates.append((item['name'], item.get('score'), '副露'))
    for code in evidence.get('dora_indicators') or []:
        candidates.append((code, 1.0, '宝牌指示牌'))

    accepted = Counter()
    excluded_low = 0
    capped = 0
    invalid = 0
    for code, score, source in candidates:
        if not isinstance(code, str) or not TILE_RE.fullmatch(code):
            invalid += 1
            continue
        try:
            confident = float(score) >= min_score
        except (TypeError, ValueError):
            confident = False
        if not confident:
            excluded_low += 1
            continue
        index = tile_index(code)
        if accepted[index] >= max(0, 4 - hand_counts[index]):
            capped += 1
            continue
        accepted[index] += 1

    def canonical(index):
        number = index % 9 + 1 if index < 27 else index - 26
        return str(number) + ('mps'[index // 9] if index < 27 else 'z')

    codes = []
    for index in sorted(accepted):
        codes.extend([canonical(index)] * accepted[index])
    audit = {
        'count': len(codes), 'by_tile': {canonical(i): n for i, n in sorted(accepted.items())},
        'excluded_low_confidence': excluded_low, 'capped_by_physical_limit': capped,
        'invalid': invalid, 'min_score': min_score,
        'note': '已见牌仅统计本地高分候选；低置信牌河/副露未计入，避免把误读硬扣为确定枚数',
    }
    return codes, audit


class ExactSolver:
    """Runs the existing audited core inside packaged V8; no Node installation."""
    def __init__(self):
        self._context = None
        self._lock = threading.Lock()

    def analyze(self, codes, seen=None):
        if len(codes) not in (13, 14):
            raise ValueError('本地核算目前需要门清的13或14张手牌')
        with self._lock:
            if self._context is None:
                from py_mini_racer import MiniRacer
                core = re.search(r'<script id="mj-core">([\s\S]*?)</script>', (ROOT / '雀魂友人场助手.html').read_text(encoding='utf-8')).group(1)
                self._context = MiniRacer()
                self._context.eval(core)
                self._context.eval('function desktopSolve(hand, seen) { return JSON.stringify(MJ.analyze(MJ.parseTiles(hand), {seen: MJ.parseTiles(seen || "")})); }')
            return json.loads(self._context.call('desktopSolve', ''.join(codes), ''.join(seen or [])))

    def close(self):
        if self._context is not None:
            self._context.close()
            self._context = None
