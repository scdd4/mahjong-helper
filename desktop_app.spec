# Build with the project's documented Python; no credentials or user screenshots are packaged.
from pathlib import Path
import os
import sys
from PyInstaller.utils.hooks import collect_all,copy_metadata

project=Path(SPECPATH)
sys.path.insert(0,str(project/'.python-deps'))
data=[(str(project/'雀魂友人场助手.html'),'.'),
      (str(project/'auto_vision_vendor'/'templates'),'auto_vision_vendor/templates'),
      (str(project/'auto_vision_vendor'/'LICENSE'),'auto_vision_vendor'),
      (str(project/'auto_vision_vendor'/'SOURCE.json'),'auto_vision_vendor')]
binaries=[];hidden=[]
for package in ('cv2','py_mini_racer'):
    package_data,package_binaries,package_hidden=collect_all(package)
    data+=package_data;binaries+=package_binaries;hidden+=package_hidden
data+=copy_metadata('mini-racer')
data+=copy_metadata('mahjong')
analysis=Analysis([str(project/'desktop_app.py')],pathex=[str(project),str(project/'.python-deps')],
                  binaries=binaries,datas=data,hiddenimports=hidden+['PIL.ImageTk','PIL.ImageGrab','scene_recognize','table_viewport','desktop_widgets','table_strategy','strategy_context','scoreboard_view','tile_recognizer','open_efficiency','table_watcher','auto_vision_vendor.meld'],
                  hookspath=[],hooksconfig={},runtime_hooks=[],excludes=['pytest','matplotlib','IPython','tkinter.test'])
archive=PYZ(analysis.pure)
exe=EXE(archive,analysis.scripts,[],exclude_binaries=True,name='MahjongStrategyAssistant',debug=False,
        bootloader_ignore_signals=False,strip=False,upx=False,console=False,disable_windowed_traceback=False,
        icon=str(project/'assets'/'app.ico'))
collection=COLLECT(exe,analysis.binaries,analysis.datas,strip=False,upx=False,name='MahjongStrategyAssistant')
