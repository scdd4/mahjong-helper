"""Standalone Windows Mahjong vision-and-strategy application."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import queue
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from desktop_runtime import (ROOT, ExactSolver, Settings, acquire_single_instance, build_seen,
                             load_settings, log_event, save_settings, tile_name)
from spark_strategy import Cancelled, SparkClient, validate_report

APP_VERSION = '1.6.1'
PHASE_LABELS = {'local':'正在识别手牌与整张牌桌','vision_ready':'识牌完成，本地规则核算就绪',
                'structured':'结构化局面已生成，模型正在推理决策',
                'vision':'Spark 正在读取牌河、副露与场况','thinking':'Spark 正在思考攻守',
                'writing':'Spark 正在生成策略','retry':'连接中断，2秒后自动重试一次'}

# Windowed executables have no console streams; some native libraries still log.
if sys.stdout is None:
    import os
    sys.stdout=open(os.devnull,'w',encoding='utf-8')
if sys.stderr is None:
    import os
    sys.stderr=open(os.devnull,'w',encoding='utf-8')


def compact_calculations(result):
    def code(index):
        if index < 27:
            return str(index % 9 + 1) + 'mps'[index // 9]
        return str(index - 26) + 'z'
    return {'shanten':result['shanten'],'win':result['win'],
            'candidates':[{'tile':code(item['tile']),'shanten':item['shanten'],
                           'ukeire':item.get('ukeireTotal'),'kinds':item.get('ukeireKinds')} for item in result['discards'][:8]],
            'waits':[{'tile':code(item['tile']),'remaining':item['remaining']} for item in result.get('waits',[])],
            'scope':'仅门清牌效率核算，不含完整攻守、场况或点数，不应据此覆盖模型策略'}


def audit_numeric_claims(report,exact):
    recommendation=report['strategy']['recommendation']
    choice=next((x for x in (exact or {}).get('candidates',[]) if x['tile']==recommendation.get('tile')),None)
    warnings=[]
    if choice:
        for number in re.findall(r'进张(?:为|有|共|约|：|:)?\s*(\d+)\s*种',str(recommendation.get('reason',''))):
            if int(number)!=choice.get('kinds'):
                warnings.append('模型对进张种数的表述与本地核算不同，以本地核算面板为准。')
    return warnings


class AnalysisPipeline:
    def __init__(self):
        self.solver = ExactSolver()
        self.client = None
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()
        if self.client:
            self.client.cancel()

    def analyze(self, image, settings, progress=None, history=None):
        from scene_recognize import read_scene
        started = time.perf_counter()
        if progress:
            progress({'phase':'local'})
        local = read_scene(image, self.cancelled)
        if self.cancelled.is_set():
            raise Cancelled('分析已取消')
        seen, seen_audit = build_seen([tile['name'] for tile in local['hand']['tiles']], local) if local['hand']['reliable'] else ([], {'count': 0, 'note': '手牌未核验，未计算已见牌'})
        exact = None
        codes = [tile['name'] for tile in local['hand']['tiles']]
        if local['hand'].get('open_hand'):
            # The JS core only computes closed 13/14 shanten; open hands rely on
            # the model with the structured meld evidence instead.
            exact = {'seen': {**seen_audit, 'count': len(seen)}, 'scope': '副露手牌：本地无门清向听核算，向听与打牌由模型结合副露推理'}
        elif local['hand']['reliable']:
            if len(codes) in (13, 14):
                try:
                    exact = compact_calculations(self.solver.analyze(codes, seen))
                    exact['seen'] = seen_audit
                except ValueError:
                    # A misread candidate must not block analysis; fall back to wall-only reference.
                    exact = compact_calculations(self.solver.analyze(codes))
                    exact['seen'] = {**seen_audit, 'count': 0, 'note': '候选已见牌与手牌冲突，本次按牌山全量计算'}
            else:
                exact = None
        if progress:
            progress({'phase':'vision_ready','hand_count':local['hand']['count']})
        self.client = SparkClient(replace(settings))
        try:
            if getattr(settings, 'analysis_mode', 'structured') == 'structured':
                result = self.client.analyze_structured(image, local, exact, history, progress)
            elif getattr(settings, 'analysis_mode', 'structured') == 'fast':
                result = self.client.analyze_fast(image, local, exact, history, progress)
            else:
                result = self.client.analyze(image, local, exact, history, progress)
        finally:
            self.client = None
        result['local_evidence'] = local
        result['exact_calculations'] = exact
        result['seen_evidence'] = seen_audit
        result['validation_warnings'] = validate_report(result['report'], local, exact)
        result['numeric_warnings']=audit_numeric_claims(result['report'],exact)
        result['total_seconds'] = round(time.perf_counter()-started,2)
        result['stage_seconds'] = {'local': round(local.get('elapsed_ms',0)/1000,2),
                                   **result.get('stage_seconds',{})}
        result['stage_seconds']['pipeline_other'] = round(max(0,result['total_seconds']-sum(result['stage_seconds'].values())),2)
        return result


def _text(value):
    if value is None:
        return '未确认'
    if isinstance(value,str):
        return value
    if isinstance(value,list):
        return '\n'.join('• '+_text(item) for item in value) or '暂无'
    if isinstance(value,dict):
        return '；'.join('%s：%s' % (k,_text(v)) for k,v in value.items())
    return str(value)


def report_sections(result):
    report = result['report']
    observed, strategy = report['observed'],report['strategy']
    action = strategy.get('recommendation',{})
    tile = action.get('tile')
    title = ('建议打 '+tile_name(tile)) if tile and tile not in ('null','unknown','无') else action.get('action','查看策略')
    warnings = result.get('validation_warnings',[])
    if warnings:
        title = '行动建议待核对'
    scores = observed.get('scores',{}) or {}
    hand = observed.get('hand',[])
    observed_text = '手牌：'+ ' '.join(tile_name(t) for t in hand)
    observed_text += '\n局数：%s    自风：%s    余牌：%s' % (_text(observed.get('round')),_text(observed.get('self_wind')),_text(observed.get('remaining')))
    if scores:
        labels={'own':'自己','right':'下家','across':'对家','left':'上家'}
        observed_text += '\n点棒：'+'  '.join(labels.get(k,k)+' '+_text(v) for k,v in scores.items())
    observed_text += '\n宝牌指示牌：'+ ' '.join(tile_name(t) for t in (observed.get('dora_indicators') or []))
    observed_text += '\n实际宝牌：'+ ' '.join(tile_name(t) for t in (observed.get('dora_tiles') or []))
    meld_lines=[]
    for seat,groups in (observed.get('melds') or {}).items():
        seat_name={'own':'本家','right':'下家','across':'对家','left':'上家'}.get(seat,seat)
        descriptions=[]
        for group in groups or []:
            if isinstance(group,dict):
                kind={'chi':'吃','pon':'碰','kan':'杠','unknown':'牌组待核对'}.get(group.get('type'),'牌组')
                cards=group.get('tiles',[])
                if group.get('confidence')=='low':kind+='（低置信）'
            else:
                kind='副露候选';cards=group
            descriptions.append(kind+' '+' '.join(tile_name(t) for t in cards))
        meld_lines.append(seat_name+'：'+('；'.join(descriptions) or '未见副露'))
    observed_text += '\n各家副露（模型读图）：\n'+'\n'.join(meld_lines)
    sections = []
    if warnings:
        sections.append(('识别核验', '\n'.join(warnings)))
    if result.get('numeric_warnings'):
        sections.append(('数字核对', '\n'.join(result['numeric_warnings'])))
    if result.get('format_repaired'):
        sections.append(('输出格式','模型的JSON语法已由本地解析器修复，未增补局面事实；请核对策略内容。'))
    if result.get('vision_uncertainties'):
        sections.append(('画面不确定信息',_text(result['vision_uncertainties'])))
    sections.extend([
        ('当前取舍 · '+_text(strategy.get('stance')), _text(action.get('reason'))),
        ('进攻路线',_text(strategy.get('offense'))),
        ('防守与风险',_text(strategy.get('defense'))),
        ('备选打法',_text(strategy.get('alternatives',[]))),
        ('后续几巡',_text(strategy.get('next_turn_plan',[]))),
        ('吃碰杠与立直',_text(strategy.get('call_plan'))),
        ('顺位与点差',_text(strategy.get('placement'))),
        ('融合局面 · 手牌来源：'+result.get('hand_source','本地核验'),observed_text),
        ('不确定信息',_text(strategy.get('uncertainties',[]))),
    ])
    exact = result.get('exact_calculations')
    if exact and 'shanten' in exact:
        sections.append(('本地核算参考','向听：%s；候选进张：%s。此项独立于模型攻守判断。' % (exact['shanten'],
                         '、'.join(tile_name(t['tile'])+' '+str(t['ukeire'])+'张' for t in exact.get('candidates',[])[:5]))))
    elif exact and exact.get('scope'):
        sections.append(('本地核算参考',str(exact['scope'])))
    return title,sections


class StrategyApp:
    BG='#101621'; PANEL='#1a2333'; TEXT='#e7edf5'; MUTED='#9aa9bf'; GOLD='#e8b45c'; RED='#ed9c91'

    def __init__(self, root):
        self.root=root
        self.root.title('雀魂策略助手')
        self.root.geometry('1220x840')
        self.root.minsize(950,650)
        self.root.configure(bg=self.BG)
        self.settings_warning = ''
        try:
            self.settings=load_settings()
        except Exception as error:
            self.settings=Settings()
            self.settings_warning = '配置读取失败，已用默认设置：%s；请重新填写模型地址与密钥' % type(error).__name__
            log_event('settings load failed: ' + type(error).__name__, 'ERROR')
        self.pipeline=AnalysisPipeline()
        self.messages=queue.Queue()
        self.busy=False;self.closed=False;self.generation=0
        self.image=None;self.preview=None;self.result=None;self.history=[]
        self.started=0;self.last_phase=''
        self.auto_enabled=False;self.watcher=None
        self._build()
        self.root.after(100,self._poll)
        self.root.protocol('WM_DELETE_WINDOW',self.close)

    def _build(self):
        style=ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame',background=self.BG)
        style.configure('TLabel',background=self.BG,foreground=self.TEXT,font=('Microsoft YaHei UI',10))
        style.configure('Primary.TButton',font=('Microsoft YaHei UI',12,'bold'),padding=(18,10),background=self.GOLD,foreground='#131924')
        style.map('Primary.TButton',background=[('active','#f2c77f'),('disabled','#5b6472')])
        menu=tk.Menu(self.root)
        file_menu=tk.Menu(menu,tearoff=False)
        file_menu.add_command(label='打开截图并分析…',command=self.open_image)
        file_menu.add_command(label='导出当前报告…',command=self.export_report)
        file_menu.add_separator();file_menu.add_command(label='退出',command=self.close)
        menu.add_cascade(label='文件',menu=file_menu)
        menu.add_command(label='模型与截图设置',command=self.show_settings)
        self.auto_var=tk.BooleanVar(value=False)
        menu.add_checkbutton(label='自动分析牌面变化',variable=self.auto_var,command=self.toggle_auto)
        menu.add_separator()
        menu.add_command(label='关于',command=self.show_about)
        menu.add_command(label='使用说明',command=lambda:messagebox.showinfo('使用说明','打开雀魂后点“截图并分析策略”。\n\n默认结构化模式：本地程序识别牌面并核算规则，模型只看结构化局面做推理决策、不识图，约7–9秒，仍开启思考。完整模式约85–148秒。\n\n菜单“自动分析牌面变化”开启后，程序只通过截图监测牌面是否稳定变化，变化后自动给出建议；可随时关闭。全程不读游戏内存、不拦截网络、不操作游戏。\n\n模型可能看错牌。高亮警告表示核验未通过。',parent=self.root))
        self.root.config(menu=menu)
        header=tk.Frame(self.root,bg=self.BG)
        header.pack(fill='x',padx=24,pady=(20,12))
        tk.Label(header,text='雀魂策略助手',font=('Microsoft YaHei UI',23,'bold'),bg=self.BG,fg=self.GOLD).pack(side='left')
        self.badge=tk.Label(header,text='',font=('Microsoft YaHei UI',10),bg=self.BG,fg=self.MUTED)
        self.badge.pack(side='right')
        toolbar=tk.Frame(self.root,bg=self.BG)
        toolbar.pack(fill='x',padx=24,pady=(0,12))
        self.button=ttk.Button(toolbar,text='截图并分析策略',style='Primary.TButton',command=self.capture)
        self.button.pack(side='left')
        self.cancel_button=ttk.Button(toolbar,text='取消分析',command=self.cancel)
        self.status=tk.Label(toolbar,text='准备就绪',bg=self.BG,fg=self.MUTED,font=('Microsoft YaHei UI',10),anchor='w')
        self.status.pack(side='left',padx=18)
        body=tk.PanedWindow(self.root,orient='horizontal',bg=self.BG,sashwidth=8,bd=0)
        body.pack(fill='both',expand=True,padx=24,pady=(0,12))
        left=tk.Frame(body,bg=self.PANEL)
        body.add(left,minsize=280,width=360)
        tk.Label(left,text='本次牌桌',bg=self.PANEL,fg=self.MUTED,font=('Microsoft YaHei UI',11)).pack(anchor='w',padx=16,pady=(16,8))
        self.image_label=tk.Label(left,text='截图后在这里预览\n也可从文件菜单打开图片',bg=self.PANEL,fg=self.MUTED,font=('Microsoft YaHei UI',11),wraplength=300)
        self.image_label.pack(fill='both',expand=True,padx=10,pady=10)
        self.local_label=tk.Label(left,text='观察整张牌桌，结合手牌、牌河、\n副露、宝牌、点棒与巡目分析。',bg=self.PANEL,fg=self.MUTED,font=('Microsoft YaHei UI',10),wraplength=325,justify='left')
        self.local_label.pack(fill='x',padx=16,pady=16)
        right=tk.Frame(body,bg=self.PANEL)
        body.add(right,minsize=500)
        self.title=tk.Label(right,text='等待分析',bg=self.PANEL,fg=self.GOLD,font=('Microsoft YaHei UI',24,'bold'),anchor='w',wraplength=700)
        self.title.pack(fill='x',padx=20,pady=(18,12))
        text_frame=tk.Frame(right,bg=self.PANEL)
        text_frame.pack(fill='both',expand=True,padx=(20,10),pady=(0,12))
        self.text=tk.Text(text_frame,bg=self.PANEL,fg=self.TEXT,insertbackground=self.TEXT,font=('Microsoft YaHei UI',11),wrap='word',relief='flat',borderwidth=0,padx=0,pady=0,spacing1=3,spacing3=5)
        scrollbar=ttk.Scrollbar(text_frame,command=self.text.yview)
        self.text.config(yscrollcommand=scrollbar.set)
        self.text.pack(side='left',fill='both',expand=True);scrollbar.pack(side='right',fill='y')
        self.text.tag_configure('heading',foreground=self.GOLD,font=('Microsoft YaHei UI',12,'bold'),spacing1=15,spacing3=6)
        self.text.tag_configure('warning',foreground=self.RED)
        self._set_text([('完整策略分析','点击后由视觉模型读取整局，给出进攻、防守、备选打法和后续计划。\n\n本地算法负责识牌候选与数值核对，策略结论来自启用思考的 Spark 模型。')])
        tk.Label(self.root,text='仅提供可见信息下的策略参考；不读取游戏内存、不操作游戏。截图将发送至所配置的模型服务。',bg=self.BG,fg='#7f8da2',font=('Microsoft YaHei UI',9)).pack(anchor='w',padx=26,pady=(0,12))
        self._update_badge()

    def _update_badge(self):
        source='雀魂窗口' if self.settings.capture_source=='window' else '全屏'
        current = getattr(self.settings,'analysis_mode','structured')
        mode='结构化' if current=='structured' else ('快速' if current=='fast' else '完整')
        badge = 'Spark · 思考 low' if current!='full' else 'Spark · 思考 '+self.settings.reasoning_effort
        badge += ' · '+source+' · '+mode+'模式'
        if self.settings_warning:
            badge = '配置异常 · ' + badge
        self.badge.config(text=badge,fg=self.RED if self.settings_warning else self.MUTED)
        if self.settings_warning:
            self._set_text([('配置需要检查',self.settings_warning+'。截图将不会发送，直到重新保存设置。')])

    def _set_text(self,sections):
        self.text.config(state='normal');self.text.delete('1.0','end')
        for heading,body in sections:
            self.text.insert('end',heading+'\n','heading')
            self.text.insert('end',body+'\n\n','warning' if heading=='识别核验' else '')
        self.text.config(state='disabled');self.text.yview_moveto(0)

    def _begin(self):
        if self.busy:
            return False
        log_event('analysis started, source=%s' % self.settings.capture_source)
        if self.settings_warning:
            messagebox.showerror('配置需要检查',self.settings_warning,parent=self.root)
            self.show_settings()
            return False
        if not self.settings.api_key:
            self.show_settings()
            if not self.settings.api_key:
                return False
        self.busy=True;self.generation+=1;self.started=time.perf_counter();self.result=None
        self.pipeline.cancelled.clear()
        self.last_phase='正在获取画面'
        if self.image:self.image.close();self.image=None
        self.preview=None;self.image_label.config(image='',text='正在获取新画面…')
        self.button.config(state='disabled');self.cancel_button.pack(side='left',padx=(12,0))
        self.title.config(text='正在分析…',fg=self.GOLD)
        self._set_text([('正在处理','正在获取新画面，旧策略已清除。')])
        self.status.config(text='正在获取画面…',fg=self.MUTED)
        return True

    def capture(self):
        if not self._begin():
            return
        generation=self.generation;settings=replace(self.settings)
        if settings.capture_source=='screen':
            self.root.withdraw()
            self.root.after(250,lambda:self._spawn_capture(generation,settings))
        else:
            self._spawn_capture(generation,settings)

    def _hide_for_capture(self):
        """Withdraw the assistant from the main thread and wait until done."""
        done=threading.Event()
        def hide():
            try:
                self.root.withdraw()
            finally:
                done.set()
        self.root.after(0,hide)
        done.wait(2.0)

    def _fullscreen_capture(self):
        """Full-screen grab with the game client bbox attached for local ROI."""
        from PIL import ImageGrab
        image=ImageGrab.grab(all_screens=True).convert('RGB')
        try:
            import ctypes
            from window_capture import Win32Backend,_select_window
            backend=Win32Backend()
            with backend.dpi_context():
                game=_select_window(backend.list_windows())
            origin_x=ctypes.windll.user32.GetSystemMetrics(76)
            origin_y=ctypes.windll.user32.GetSystemMetrics(77)
            image.info['game_bbox']=[game.client_left-origin_x,game.client_top-origin_y,
                                    game.client_left-origin_x+game.client_width,game.client_top-origin_y+game.client_height]
        except Exception:
            pass
        return image

    def _spawn_capture(self,generation,settings):
        def work():
            try:
                if settings.capture_source=='window':
                    from window_capture import capture_mahjong_window,WindowCaptureError
                    image=None
                    try:
                        image,_=capture_mahjong_window()
                    except WindowCaptureError as error:
                        # Unity/D3D games often return a blank PrintWindow frame and
                        # reject screen-copy when any overlay sits above them. Fall
                        # back to a composited full-screen grab, cropped by game bbox.
                        if error.code in ('window_occluded','blank_frame','capture_unavailable',
                                          'window_not_visible','capture_denied'):
                            log_event('window capture fallback: '+error.code)
                            self.messages.put((generation,'notice','窗口截图受遮挡，已自动改用全屏截图'))
                            self._hide_for_capture()
                            image=self._fullscreen_capture()
                            image.info['capture_fallback']=error.code
                            if not image.info.get('game_bbox'):
                                raise WindowCaptureError('window_not_visible',
                                    '窗口截图受遮挡，且全屏画面中未定位到雀魂客户区。请把雀魂窗口恢复并完整露出后重试。')
                        else:
                            raise
                else:
                    image=self._fullscreen_capture()
                self.messages.put((generation,'restore',None))
                self._work_image(image,settings,generation)
            except Exception as error:
                self.messages.put((generation,'restore',None))
                self.messages.put((generation,'error',str(error)))
        threading.Thread(target=work,daemon=True).start()

    def open_image(self):
        if self.busy:
            return
        path=filedialog.askopenfilename(title='打开牌桌截图',filetypes=[('截图','*.png *.jpg *.jpeg *.webp *.bmp'),('所有文件','*.*')],parent=self.root)
        if path:
            self.analyze_file(path)

    def analyze_file(self,path):
        if not self._begin():
            return
        generation=self.generation;settings=replace(self.settings)
        def work():
            try:
                from PIL import Image
                with Image.open(path) as source:
                    image=source.convert('RGB')
                self._work_image(image,settings,generation)
            except Exception as error:
                self.messages.put((generation,'error',str(error)))
        threading.Thread(target=work,daemon=True).start()

    def _work_image(self,image,settings,generation):
        preview=image.copy();preview.info=dict(image.info)
        self.messages.put((generation,'image',preview))
        try:
            result=self.pipeline.analyze(image,settings,lambda event:self.messages.put((generation,'progress',event)),self.history[-3:])
            self.messages.put((generation,'result',result))
        except Cancelled:
            self.messages.put((generation,'error','分析已取消'))
        except Exception as error:
            self.messages.put((generation,'error',str(error)))
        finally:
            image.close()

    def show_about(self):
        messagebox.showinfo('关于','雀魂策略助手 v%s\n\n'
            '输入仅为程序截图：本地识别牌面与规则核算，结构化局面交给模型推理决策。\n'
            '不读游戏内存、不拦截网络、不操作游戏。截图会发送到所配置的模型服务。\n\n'
            '模式：结构化（约8秒，推荐）／快速识图（约9秒）／完整（85秒以上）。'
            % APP_VERSION,parent=self.root)

    def _grab_for_watch(self):
        """Screenshot only. Returns a fresh image or None; never touches the game."""
        try:
            if self.busy or self.closed or self.settings.capture_source!='window':
                return None
            from window_capture import capture_mahjong_window
            image,_=capture_mahjong_window()
            return image
        except Exception:
            return None

    def toggle_auto(self):
        from table_watcher import TableWatcher
        if self.auto_var.get():
            if self.settings_warning or not self.settings.api_key:
                self.auto_var.set(False)
                messagebox.showinfo('自动分析','请先配置模型密钥，再开启自动分析。',parent=self.root)
                return
            self.auto_enabled=True
            self.watcher=TableWatcher(self._grab_for_watch,self._auto_trigger)
            self.watcher.start()
            self.watcher.reset()
            self.status.config(text='自动分析已开启：牌面稳定变化后自动给出建议',fg='#8bcdb1')
            log_event('auto analysis on')
        else:
            self.auto_enabled=False
            if self.watcher:
                self.watcher.stop();self.watcher=None
            self.status.config(text='准备就绪',fg=self.MUTED)
            log_event('auto analysis off')

    def _auto_trigger(self):
        """Called from watcher thread; hop to Tk thread and reuse manual capture."""
        if self.closed or self.busy or not self.auto_enabled:
            return
        self.root.after(0,self.capture)

    def show_image(self,image):
        from PIL import Image,ImageTk
        if self.image:
            self.image.close()
        self.image=image.copy()
        preview=image.copy();preview.thumbnail((340,480),Image.Resampling.LANCZOS)
        self.preview=ImageTk.PhotoImage(preview)
        self.image_label.config(image=self.preview,text='');preview.close()

    def show_result(self,result):
        self.result=result
        title,sections=report_sections(result)
        self.title.config(text=title,fg=self.RED if result.get('validation_warnings') else self.GOLD)
        self._set_text(sections)
        local=result.get('local_evidence',{}).get('hand',{})
        self.local_label.config(text='本地手牌：'+ ' '.join(tile_name(t.get('name') or '?') for t in local.get('tiles',[]))+'\n'+('本地手牌核验通过' if local.get('reliable') else '本地手牌尚待核验'))
        elapsed=result.get('total_seconds',result.get('elapsed_seconds',0))
        stages=result.get('stage_seconds',{})
        detail=' · '.join('%s %.1fs' % (name,value) for name,value in stages.items() if isinstance(value,(int,float)))
        effort = 'low' if result.get('mode') in ('fast','structured') else result.get('reasoning_effort','medium')
        prefix={'fast':'快速','structured':'结构化'}.get(result.get('mode'),'完整')
        self.status.config(text=prefix+' · 分析完成 · %.1f 秒 · 思考 %s%s' % (elapsed,effort,(' · '+detail) if detail else ''),fg='#8bcdb1')
        observed=result['report']['observed'];strategy=result['report']['strategy']
        round_label=_text(observed.get('round')); self_wind=_text(observed.get('self_wind'))
        remaining=observed.get('remaining')
        previous=self.history[-1] if self.history else {}
        # New hand when the round/wind label changes or the wall counter jumps
        # back up (a fresh deal), not on the normal per-turn decrease.
        def numeric(value):
            try:
                return int(value)
            except (TypeError,ValueError):
                return None
        new_hand = (previous.get('round') not in (None,round_label) or previous.get('self_wind') not in (None,self_wind)
                    or (numeric(remaining) is not None and numeric(previous.get('remaining')) is not None
                        and numeric(remaining) > numeric(previous.get('remaining')) + 2))
        if new_hand:
            self.history=[]
        self.history.append({'round':round_label,'self_wind':self_wind,'remaining':remaining,
                             'stance':strategy.get('stance'),'recommendation':strategy.get('recommendation'),
                             'hand_source':result.get('hand_source')})
        self.history=self.history[-5:]
        self._finish()

    def _finish(self):
        self.busy=False;self.button.config(state='normal');self.cancel_button.pack_forget()
        if self.watcher:
            self.watcher.reset()  # re-baseline so the just-analysed state doesn't retrigger

    def _poll(self):
        if self.closed:
            return
        while True:
            try:
                generation,kind,value=self.messages.get_nowait()
            except queue.Empty:
                break
            if generation!=self.generation:
                if kind=='image':
                    value.close()
                continue
            if kind=='restore':
                self.root.deiconify()
            elif kind=='image':
                self.show_image(value);value.close()
            elif kind=='notice':
                self.status.config(text=value,fg=self.GOLD)
            elif kind=='progress':
                self.last_phase=PHASE_LABELS.get(value.get('phase'),'正在分析')
            elif kind=='result':
                self.show_result(value)
                log_event('analysis completed in %ss' % value.get('total_seconds','?'))
            elif kind=='error':
                log_event('analysis failed: ' + str(value)[:120].replace('\n',' '), 'ERROR')
                self.title.config(text='本次分析未完成',fg=self.RED)
                self._set_text([('原因',value),('下一步','可在模型与截图设置中检查连接，或等待牌面稳定后重试。')])
                self.status.config(text=value[:90],fg=self.RED);self._finish()
        if self.busy:
            self.status.config(text=self.last_phase+' · 已等待 %d 秒' % (time.perf_counter()-self.started))
        self.root.after(100,self._poll)

    def cancel(self):
        self.pipeline.cancel()
        self.status.config(text='正在取消…')

    def export_report(self):
        if not self.result:
            messagebox.showinfo('导出报告','先完成一次分析。',parent=self.root);return
        path=filedialog.asksaveasfilename(title='导出策略报告',defaultextension='.txt',filetypes=[('文字报告','*.txt'),('结构化报告','*.json')],parent=self.root)
        if not path:
            return
        if Path(path).suffix.lower()=='.json':
            text=json.dumps(self.result,ensure_ascii=False,indent=2)
        else:
            title,sections=report_sections(self.result)
            text=title+'\n\n'+'\n\n'.join(h+'\n'+b for h,b in sections)
        Path(path).write_text(text,encoding='utf-8')

    def show_settings(self):
        if self.busy:
            messagebox.showinfo('正在分析','请先等待或取消当前分析，再修改模型和截图设置。',parent=self.root);return
        dialog=tk.Toplevel(self.root);dialog.title('模型与截图设置');dialog.geometry('600x640');dialog.resizable(False,False)
        dialog.transient(self.root);dialog.grab_set()
        frame=ttk.Frame(dialog,padding=22);frame.pack(fill='both',expand=True)
        entries={}
        for row,(field,label) in enumerate([('base_url','Base URL'),('model','模型名称'),('api_key','API 密钥')]):
            ttk.Label(frame,text=label).grid(row=row*2,column=0,sticky='w',pady=(5,3))
            variable=tk.StringVar(value=getattr(self.settings,field));entries[field]=variable
            ttk.Entry(frame,textvariable=variable,width=65,show='•' if field=='api_key' else '').grid(row=row*2+1,column=0,columnspan=2,sticky='ew',pady=(0,9))
        values={}
        for row,(field,label,options) in enumerate([('reasoning_effort','思考强度',('low','medium','high')),('max_tokens','输出上限',('8192','16384','32768')),('timeout','超时（秒）',('600','900','1200'))],start=6):
            ttk.Label(frame,text=label).grid(row=row,column=0,sticky='w',pady=7)
            variable=tk.StringVar(value=str(getattr(self.settings,field)));values[field]=variable
            ttk.Combobox(frame,textvariable=variable,values=options,state='readonly',width=16).grid(row=row,column=1,sticky='e')
        source=tk.StringVar(value='雀魂窗口' if self.settings.capture_source=='window' else '全屏')
        ttk.Label(frame,text='截图来源').grid(row=9,column=0,sticky='w',pady=8)
        ttk.Combobox(frame,textvariable=source,values=('雀魂窗口','全屏'),state='readonly',width=16).grid(row=9,column=1,sticky='e')
        current_mode = getattr(self.settings,'analysis_mode','structured')
        mode_labels={'structured':'结构化（约7秒，推荐）','fast':'快速识图（约9秒）','full':'完整（85秒以上）'}
        mode_var=tk.StringVar(value=mode_labels.get(current_mode,mode_labels['structured']))
        ttk.Label(frame,text='分析模式').grid(row=10,column=0,sticky='w',pady=8)
        ttk.Combobox(frame,textvariable=mode_var,values=tuple(mode_labels.values()),state='readonly',width=20).grid(row=10,column=1,sticky='e')
        ttk.Label(frame,text='结构化：本地程序识别+规则核算，模型只看结构化局面做决策，约7秒。\n快速识图：模型看一张压缩图辅助判断，约9秒。完整：两阶段深度分析，85秒以上。\n三种模式都开启思考；不读游戏内存、不拦截网络。',wraplength=540).grid(row=11,column=0,columnspan=2,sticky='w',pady=(2,8))
        ttk.Label(frame,text='密钥仅保存在当前 Windows 用户的加密配置中。\n截图会发送到以上模型地址；全屏模式包含其他可见窗口。',wraplength=540).grid(row=12,column=0,columnspan=2,sticky='w',pady=(14,10))
        def save():
            try:
                reverse_mode={v:k for k,v in mode_labels.items()}
                settings=Settings(**{k:v.get().strip() for k,v in entries.items()},reasoning_effort=values['reasoning_effort'].get(),
                                  max_tokens=int(values['max_tokens'].get()),timeout=int(values['timeout'].get()),
                                  capture_source='window' if source.get()=='雀魂窗口' else 'screen',
                                  analysis_mode=reverse_mode.get(mode_var.get(),'structured'))
                save_settings(settings);self.settings=settings;self.settings_warning=''
                self._update_badge();log_event('settings saved');dialog.destroy()
            except Exception as error:
                messagebox.showerror('设置未保存',str(error),parent=dialog)
        ttk.Button(frame,text='保存设置',command=save).grid(row=13,column=1,sticky='e',pady=8)
        frame.columnconfigure(0,weight=1)
        self.root.wait_window(dialog)

    def close(self):
        self.closed=True;self.generation+=1;self.pipeline.cancel()
        if self.watcher:
            self.watcher.stop();self.watcher=None
        if self.image:
            self.image.close()
        self.root.destroy()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image')
    parser.add_argument('--show-report')
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--check-output')
    parser.add_argument('--single-instance-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        from PIL import Image
        import cv2
        settings=load_settings()
        solver=ExactSolver();result=solver.analyze(['1m','2m','3m','4m','5m','6m','7m','8m','9m','2p','3p','4p','5p','4z']);solver.close()
        text=json.dumps({'ok':result['total']==14,'model_configured':bool(settings.api_key),'thinking':settings.reasoning_effort,'opencv':cv2.__version__})
        if args.check_output:
            Path(args.check_output).write_text(text,encoding='utf-8')
        elif sys.stdout:
            print(text)
        return
    if args.single_instance_check:
        ok = acquire_single_instance()
        print(json.dumps({'first_acquired':ok}))
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    try:
        if not acquire_single_instance():
            messagebox.showerror('已在运行','雀魂策略助手已在运行，请切换到已有窗口。')
            return
    except Exception as error:
        messagebox.showerror('启动失败','无法创建单实例锁：%s' % error)
        return
    log_event('application started')
    root=tk.Tk();app=StrategyApp(root)
    if args.show_report:
        result=json.loads(Path(args.show_report).read_text(encoding='utf-8'))
        root.after(200,lambda:app.show_result(result))
        if args.image:
            from PIL import Image
            with Image.open(args.image) as source:
                app.show_image(source.convert('RGB'))
    elif args.image:
        root.after(300,lambda:app.analyze_file(args.image))
    root.mainloop()


if __name__=='__main__':
    main()
