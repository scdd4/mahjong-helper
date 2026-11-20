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
import uuid
import time
from datetime import datetime

ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
_INSTANCE_MUTEX = None
LOG_NAME = 'diagnostics.log'
if (ROOT / '.python-deps').is_dir():
    sys.path.insert(0, str(ROOT / '.python-deps'))

APP_NAME = 'MahjongStrategyAssistant'
REGISTRY_SETTINGS_KEY = r'Software\MahjongStrategyAssistant'
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
    # 本地绝对路径（Windows 盘符/UNC）属于隐私，不应写入诊断日志
    text = re.sub(r'[A-Za-z]:\\[^\s\'"]+', '[local-path]', text)
    text = re.sub(r'\\\\[^\s\'"]+', '[unc-path]', text)
    try:
        target = user_directory() / LOG_NAME
        stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with target.open('a', encoding='utf-8') as stream:
            stream.write('%s [%s] %s\n' % (stamp, level, text))
    except Exception:
        if os.name=='nt':
            try:
                import winreg
                with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER,REGISTRY_SETTINGS_KEY,0,winreg.KEY_SET_VALUE) as key:
                    winreg.SetValueEx(key,'LastDiagnostic',0,winreg.REG_SZ,level+' '+text)
            except OSError:pass


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
    base_url: str = ''  # 不预置任何内网/明文地址；首次运行必须在设置里显式填写
    model: str = 'qwen3.8-flash-next-iq3_s'
    api_key: str = ''
    reasoning_effort: str = 'medium'
    max_tokens: int = 16384
    timeout: int = 600
    capture_source: str = 'window'
    analysis_mode: str = 'structured'
    mini_mode: bool = False
    mini_topmost: bool = True

    def validate(self):
        from urllib.parse import urlsplit
        url = self.base_url.strip()
        if url:  # 空串 = 尚未配置，由调用方给出明确提示，不视为合法地址
            parsed = urlsplit(url)
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
        if not isinstance(self.mini_mode, bool) or not isinstance(self.mini_topmost, bool):
            raise ValueError('窗口模式设置无效')
        return self

    @property
    def endpoint(self):
        value = self.base_url.rstrip('/')
        return value if value.endswith('/chat/completions') else value + '/chat/completions'


def _registry_settings(value=None):
    if os.name!='nt':return None
    import winreg
    if value is None:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,REGISTRY_SETTINGS_KEY) as key:
                return json.loads(winreg.QueryValueEx(key,'EncryptedSettings')[0])
        except FileNotFoundError:return None
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER,REGISTRY_SETTINGS_KEY,0,winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key,'EncryptedSettings',0,winreg.REG_SZ,json.dumps(value,ensure_ascii=False))
    return '当前用户配置存储'


def portable_settings_path():
    return Path(sys.executable).parent/'user-data'/'settings.json' if getattr(sys,'frozen',False) else None


def _write_settings_file(value,target):
    temporary = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # tempfile retries PermissionError on Windows when os.access claims
        # the directory is writable. Per-process denials can then loop for
        # hundreds of thousands of names instead of reaching our fallback.
        for attempt in range(3):
            candidate=target.parent/('settings-'+uuid.uuid4().hex+'.tmp')
            try:
                stream=candidate.open('x',encoding='utf-8')
                temporary=candidate
                break
            except FileExistsError:
                if attempt==2:raise
        with stream:
            json.dump(value,stream,ensure_ascii=False,indent=2)
        os.replace(temporary, target)
    finally:
        if temporary and temporary.exists():
            try:temporary.unlink()
            except OSError:pass
    return str(target)


def save_settings(settings, path=None):
    settings.validate()
    target=Path(path) if path else Path(os.environ.get('LOCALAPPDATA',Path.home()/'AppData'/'Local'))/APP_NAME/'settings.json'
    value = asdict(settings)
    key = value.pop('api_key')
    value['api_key_dpapi'] = base64.b64encode(_crypt(key.encode('utf-8'))).decode('ascii') if key else ''
    value['saved_at_ns']=time.time_ns()
    try:
        return _write_settings_file(value,target)
    except PermissionError as error:
        if path is None:
            try:return _registry_settings(value)
            except OSError:pass
            portable=portable_settings_path()
            if portable is not None:
                try:return _write_settings_file(value,portable)
                except OSError:pass
        raise PermissionError('当前会话不能保存设置，原设置已保留。默认目录、当前账户配置和应用目录均无法写入，请从有写入权限的位置打开助手。') from error


def load_settings(path=None):
    target=Path(path) if path else Path(os.environ.get('LOCALAPPDATA',Path.home()/'AppData'/'Local'))/APP_NAME/'settings.json'
    options=[];failure=None
    files=[target]
    if path is None and portable_settings_path() is not None:files.append(portable_settings_path())
    for candidate in files:
        if candidate.is_file():
            try:
                value=json.loads(candidate.read_text(encoding='utf-8-sig'))
                options.append((value.get('saved_at_ns',candidate.stat().st_mtime_ns),value))
            except (ValueError,OSError) as error:failure=error
    if path is None:
        try:
            value=_registry_settings()
            if value:options.append((value.get('saved_at_ns',0),value))
        except (ValueError,OSError) as error:failure=failure or error
    if not options:
        if failure:raise failure
        return Settings()
    value=dict(max(options,key=lambda entry:entry[0])[1])
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
    return ('赤' if code[0] == '0' else '') + '零一二三四五六七八九'[5 if code[0]=='0' else int(code[0])] + {'m': '万', 'p': '筒', 's': '条'}[code[1]]


def tile_code(value):
    if not isinstance(value,str):return None
    value=value.strip()
    if TILE_RE.fullmatch(value):return value
    honors={'东':'1z','南':'2z','西':'3z','北':'4z','白':'5z','发':'6z','發':'6z','中':'7z'}
    if value in honors:return honors[value]
    match=re.fullmatch(r'(赤|红)?([一二三四五六七八九1-9])([万萬筒饼餅条條索])',value)
    if not match:return None
    red,number,suit=match.groups()
    number=int(number) if number.isdigit() else '零一二三四五六七八九'.index(number)
    if red and number!=5:return None
    return ('0' if red else str(number))+({'万':'m','萬':'m','筒':'p','饼':'p','餅':'p','条':'s','條':'s','索':'s'}[suit])


def resolve_tile_choice(value,hand):
    code=tile_code(value)
    if code in hand:return code
    if code and code[0]=='5' and '0'+code[1] in hand:return '0'+code[1]
    return code


def humanize_tiles(value):
    text=str(value)
    def convert(match):
        numbers,suit=match.groups()
        return '、'.join(tile_name(n+suit) for n in numbers)
    text=re.sub(r'(?<![A-Za-z0-9])([0-9]+)([mps]|z)(?![A-Za-z0-9])',convert,text)
    return re.sub(r'([一二三四五六七八九1-9])[索條]',r'\1条',text)


def group_tiles(group):
    return group.get('tiles',[]) if isinstance(group,dict) else (group or [])


def effective_hand_total(hand, groups):
    return len(hand)+3*len(groups or [])


def dora_from_indicator(code):
    tile_index(code)
    number = 5 if code[0]=='0' else int(code[0])
    if code[1]!='z':
        return str(number%9+1)+code[1]
    # Honor dora succession is TWO independent cycles, verified against six
    # authoritative Chinese riichi references and in-game behavior:
    # winds 東→南→西→北→東, dragons 白→發→中→白. (The 東南西北白發中 order
    # is only the tile-numbering convention, not a dora cycle.)
    return str({1:2,2:3,3:4,4:1,5:6,6:7,7:5}[number])+'z'


WIND_CODE = {'东': '1z', '南': '2z', '西': '3z', '北': '4z'}
# 筋 = 与立直者切牌相差 3 的同花色数牌。两侧筋(切4/5/6)为强线索；
# 端张(1/2/3/7/8/9)也有单侧筋(x±3)，但强度更弱，单列字段不并入强筋。
SUJI_FROM_MIDDLE = {4: (1, 7), 5: (2, 8), 6: (3, 9)}
SUJI_SINGLE = {1:(4,),2:(5,),3:(6,),7:(4,),8:(5,),9:(6,)}


def build_safety_facts(rivers, riichi, round_wind=None, self_wind=None):
    """Per-player visible genbutsu and limited suji evidence, never a global safe union.

    Red fives share a tile kind. Repeated discards do not create two-sided suji;
    honors, suji and two-sided suji are not guarantees against every wait shape.
    """
    facts = {'现物': {}, '筋': {}, '单筋': {}, '两筋': {}, '客风客三元': [],
             '说明':'现物只对对应玩家成立；筋仍可能被嵌张、边张、双碰或单骑命中。'
                    '单筋(切端张产生的 x±3)比两侧筋更弱，仅作参考。三元牌都是役牌，字牌不能视为绝对安全。'}
    for seat, declared in (riichi or {}).items():
        if not declared:
            continue
        river = [t for t in (rivers.get(seat) or []) if isinstance(t, str) and TILE_RE.fullmatch(t)]
        river = [tile_name_to_canonical(t) for t in river]
        facts['现物'][seat] = sorted(set(river))
        suji, double = set(), set()
        for tile in river:
            if tile.endswith('z'):
                continue
            number = 5 if tile[0] == '0' else int(tile[0])
            for side in SUJI_FROM_MIDDLE.get(number, ()):
                code = str(side) + tile[1]
                if code not in facts['现物'][seat]:
                    suji.add(code)
        single = set()
        for tile in river:
            if tile.endswith('z'):
                continue
            number = 5 if tile[0] == '0' else int(tile[0])
            for side in SUJI_SINGLE.get(number, ()):
                code = str(side) + tile[1]
                if code not in facts['现物'][seat] and code not in suji:
                    single.add(code)
        facts['单筋'][seat] = sorted(single - set(facts['现物'][seat]))
        for suit in 'mps':
            for middle in (4,5,6):
                if str(middle-3)+suit in river and str(middle+3)+suit in river:
                    code=str(middle)+suit
                    if code not in river:double.add(code)
        facts['筋'][seat] = sorted(suji - set(facts['现物'][seat]))
        facts['两筋'][seat] = sorted(double)
    return facts


def common_safe_tiles(facts, seats, hand=None):
    """Tiles safe against *every* declared riichi player (intersection, not union).

    Uses confirmed genbutsu and strong suji from each riichi player's own river.
    Single-sided suji and honors are deliberately excluded: they are weak cues,
    not verified safety. A tile is returned only if safe for all given seats.
    """
    seats = [s for s in seats if s in (facts.get('现物') or {})]
    if not seats:
        return []
    pools = []
    for seat in seats:
        pool = set(facts['现物'].get(seat, [])) | set(facts.get('筋', {}).get(seat, [])) | set(facts.get('两筋', {}).get(seat, []))
        pools.append(pool)
    common = set.intersection(*pools)
    if hand:
        common &= set(hand)
    return sorted(common)


def tile_name_to_canonical(code):
    return '5'+code[1] if code[0]=='0' else code


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
                core_html = (ROOT / '雀魂友人场助手.html').read_text(encoding='utf-8')
                match = re.search(r'<script id="mj-core">([\s\S]*?)</script>', core_html)
                if not match:
                    raise ValueError('本地核算核心脚本缺失：雀魂友人场助手.html 缺少 mj-core 段，请检查安装完整性')
                core = match.group(1)
                self._context = MiniRacer()
                self._context.eval(core)
                self._context.eval('function desktopSolve(hand, seen) { return JSON.stringify(MJ.analyze(MJ.parseTiles(hand), {seen: MJ.parseTiles(seen || "")})); }')
            return json.loads(self._context.call('desktopSolve', ''.join(codes), ''.join(seen or [])))

    def close(self):
        if self._context is not None:
            self._context.close()
            self._context = None
