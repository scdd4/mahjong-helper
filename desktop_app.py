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
from desktop_runtime import (ROOT, ExactSolver, Settings, acquire_single_instance, build_seen,humanize_tiles,
                             load_settings, log_event, save_settings, tile_name)
from spark_strategy import Cancelled, SparkClient, validate_report

APP_VERSION = '1.10.0'
PHASE_LABELS = {'local':'正在识别手牌与整张牌桌','vision_ready':'识牌完成，本地规则核算就绪',
                'structured':'结构化局面已生成，模型正在推理决策',
                'vision':'Spark 正在读取牌河、副露与场况','thinking':'Spark 正在思考攻守',
                'writing':'Spark 正在生成策略','retry':'连接中断，2秒后自动重试一次',
                'advice':'已给出初步建议，正在补全场面分析'}

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
            'forms':{'普通型':result.get('forms',{}).get('std'),'七对子':result.get('forms',{}).get('chiitoi'),'国士':result.get('forms',{}).get('kokushi')},
            'candidates':[{'tile':code(item['tile']),'shanten':item['shanten'],
                           'ukeire':item.get('ukeireTotal'),'kinds':item.get('ukeireKinds')} for item in result['discards'][:8]],
            # 13 张时 discards 为空，有效进张在 ukeire（按剩余枚数排序），快速模式的前瞻依赖它。
            'ukeire':[{'tile':code(item['tile']),'remaining':item['remaining']} for item in result.get('ukeire',[])],
            'waits':[{'tile':code(item['tile']),'remaining':item['remaining']} for item in result.get('waits',[])],
            'scope':'仅门清牌效率核算，不含完整攻守、场况或点数，不应据此覆盖模型策略'}


def audit_numeric_claims(report,exact):
    recommendation=report['strategy']['recommendation']
    choice=next((x for x in (exact or {}).get('candidates',[]) if x['tile']==recommendation.get('tile')),None)
    warnings=[]
    text=str(recommendation.get('reason',''))
    if choice:
        for number in re.findall(r'(?:进张|有效牌|受入)(?:为|有|共|约|：|:)?\s*(\d+)\s*[种张枚面]',text):
            if int(number)!=choice.get('kinds'):
                warnings.append('模型对进张数量的表述与本地核算不同，以本地核算面板为准。')
                break
    # 退向听属于方向性错误，即使主弃牌正确也提示
    if re.search(r'退(?:一|1)?向听|向听(?:变|退)差',text):
        warnings.append('模型描述退向听，与保持最优向听的核算目标冲突，请复核。')
    return warnings


class AnalysisPipeline:
    def __init__(self):
        self.solver = ExactSolver()
        self.client = None
        self.cancelled = threading.Event()

    def close(self):
        """Release the V8 context; without this every restart leaks an instance."""
        self.cancelled.set()
        try:
            self.solver.close()
        finally:
            self.solver = None

    def cancel(self):
        self.cancelled.set()
        if self.client:
            self.client.cancel()

    def analyze(self, image, settings, progress=None, history=None):
        if self.solver is None:
            raise RuntimeError('本地核算器已关闭，无法继续分析')
        from scene_recognize import read_scene
        started = time.perf_counter()
        first_advice_at=None
        def track_progress(event):
            nonlocal first_advice_at
            if event.get('phase')=='advice' and first_advice_at is None:
                first_advice_at=round(time.perf_counter()-started,2)
            if progress:progress(event)
        if progress:
            progress({'phase':'local'})
        # 快速模式只需要手牌与立直，跳过牌河/副露的逐张识别（约省 1.2s）。
        # Jev 模式需要牌河/立直事实做攻守判断，因此用完整识别。
        mode_now = getattr(settings, 'analysis_mode', 'structured')
        local = read_scene(image, self.cancelled, fast=(mode_now == 'fast'))
        if self.cancelled.is_set():
            raise Cancelled('分析已取消')
        hand = local.get('hand') or {}
        if not hand.get('tiles'):
            raise ValueError('手牌识别为空，请让牌面完整清晰后重新截图')
        seen, seen_audit = build_seen([tile['name'] for tile in hand['tiles']], local) if hand.get('reliable') else ([], {'count': 0, 'note': '手牌未核验，未计算已见牌'})
        exact = None
        codes = [tile['name'] for tile in hand['tiles']]
        if hand.get('open_hand'):
            from open_efficiency import analyze_open
            try:
                exact=analyze_open(codes,(local.get('melds') or {}).get('own',{}).get('groups',[]),seen)
                exact['seen']=seen_audit
            except ValueError as error:
                exact=None
                log_event('open-hand local calc skipped: %s' % error, 'WARNING')
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
        if self.cancelled.is_set():
            raise Cancelled('分析已取消')
        self.client = SparkClient(replace(settings), cancel_event=self.cancelled)
        try:
            if mode_now == 'structured':
                result = self.client.analyze_structured(image, local, exact, history, track_progress)
            elif mode_now == 'jev':
                result = self.client.decide_jev(image, local, exact, history, track_progress)
            elif mode_now == 'fast':
                result = self.client.analyze_fast(image, local, exact, history, track_progress)
            else:
                result = self.client.analyze(image, local, exact, history, track_progress)
        finally:
            self.client = None
        result['local_evidence'] = local
        result['exact_calculations'] = exact
        result['seen_evidence'] = seen_audit
        result['validation_warnings'] = validate_report(result['report'], local, exact)
        result['numeric_warnings']=audit_numeric_claims(result['report'],exact)
        result['total_seconds'] = round(time.perf_counter()-started,2)
        if first_advice_at is not None:
            result['first_advice_total_seconds']=first_advice_at
        result['stage_seconds'] = {'local': round(local.get('elapsed_ms',0)/1000,2),
                                   **result.get('stage_seconds',{})}
        result['stage_seconds']['pipeline_other'] = round(max(0,result['total_seconds']-sum(result['stage_seconds'].values())),2)
        return result


def _text(value):
    if value is None:
        return '未确认'
    if isinstance(value,str):
        return humanize_tiles(value)
    if isinstance(value,list):
        return '\n'.join('• '+_text(item) for item in value) or '暂无'
    if isinstance(value,dict):
        labels={'tile':'牌','when':'改选条件','reason':'理由','type':'类型','confidence':'置信度'}
        return '；'.join('%s：%s' % (labels.get(k,k),_text(v)) for k,v in value.items())
    return str(value)


def report_sections(result):
    report = result['report']
    observed, strategy = report['observed'],report['strategy']
    action = strategy.get('recommendation',{})
    tile = action.get('tile')
    fast = bool(result.get('thinking_stages',{}).get('fast')) or result.get('mode') == 'jev'
    if tile and tile not in ('null','unknown','无'):
        title = '建议打 '+tile_name(tile)
    elif fast and strategy.get('next_discard'):
        # 未轮到：直接给"轮到你打哪张"，这比"给你前瞻"有用。
        title = '轮到你打 '+tile_name(strategy['next_discard'])
    elif action.get('action') == '等待摸牌':
        title = '未轮到你 · 给你前瞻'
    else:
        title = action.get('action','查看策略')
    warnings = result.get('validation_warnings',[])
    if warnings:
        title = '行动建议待核对'
    # 快速模式只要结论：标题即答案，正文两行内收口，不铺开场面细节。
    if fast and not warnings:
        quick = []
        reason = _text(action.get('reason'))
        if reason and reason != '未确认':
            quick.append(('结论', reason))
        # 标题已经是"建议打 X / 轮到你打 X"，正文不再重复同一动作。
        if strategy.get('defense'):
            quick.append(('要防的', _text(strategy['defense'])))
        issues = [w for w in (result.get('vision_uncertainties') or [])
                  if '模板结果不是完整场况' not in w and '程序判断可直接和牌' not in w]
        if issues:
            quick.append(('需留意', _text(issues)))
        hand = observed.get('hand',[])
        quick.append(('手牌', ' '.join(tile_name(t) for t in hand)))
        return title, quick
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
    observed_text += '\n各家副露（识别候选）：\n'+'\n'.join(meld_lines)
    seats={'own':'自己','right':'下家','across':'对家','left':'上家'}
    observed_text += '\n各家牌河（可见候选）：\n'+'\n'.join(seats.get(s,s)+'：'+(' '.join(tile_name(t) for t in tiles) or '未读到') for s,tiles in (observed.get('rivers') or {}).items())
    observed_text += '\n立直状态：'+'；'.join(seats.get(s,s)+' '+('已立直' if value is True else ('疑似立直(待复核)' if value=='candidate' else ('未立直' if value is False else '未确认'))) for s,value in (observed.get('riichi') or {}).items())
    sections = []
    if warnings:
        sections.append(('识别核验', '\n'.join(warnings)))
    if result.get('numeric_warnings'):
        sections.append(('数字核对', '\n'.join(result['numeric_warnings'])))
    if result.get('format_repaired'):
        sections.append(('输出格式','模型的JSON语法已由本地解析器修复，未增补局面事实；请核对策略内容。'))
    if result.get('rule_corrections'):
        sections.append(('规则核对',_text(result['rule_corrections'])))
    if result.get('detail_incomplete'):
        sections.append(('详解未完成','以下保留本次初步建议；可在牌面稳定后重新截图。'))
    if result.get('vision_uncertainties'):
        sections.append(('画面不确定信息',_text(result['vision_uncertainties'])))
    sections.extend([
        ('场上威胁',_text(strategy.get('table_threats',[]))),
        ('当前取舍 · '+_text(strategy.get('stance')), _text(action.get('reason'))),
        ('进攻路线',_text(strategy.get('offense'))),
        ('防守与风险',_text(strategy.get('defense'))),
        ('备选打法',_text(strategy.get('alternatives',[]))),
        (('轮到你时考虑' if result.get('thinking_stages',{}).get('fast') else '后续几巡'),
         _text(strategy.get('next_turn_plan',[]))),
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
        self.root.title('雀魂策略助手 v' + APP_VERSION)
        # 窄而高的布局：宽度收窄，高度贴近屏幕可用高度（任务栏留边），
        # 窗口贴屏幕右侧，方便与左侧雀魂窗口并排实战。
        screen_w=self.root.winfo_screenwidth()
        screen_h=self.root.winfo_screenheight()
        win_w=900
        win_h=max(760,min(screen_h-80,1080))  # 上下尽量占满，最多1080避免超屏
        pos_x=max(0,screen_w-win_w-20)
        self.default_geometry='%dx%d+%d+0' % (win_w,win_h,pos_x)
        self.root.geometry(self.default_geometry)
        self.root.minsize(820,700)
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
        self.auto_enabled=False;self.watcher=None;self.pending_capture=False;self.initial_advice=None
        self.mini_mode=False;self.normal_geometry=self.default_geometry
        self.root.report_callback_exception=self._ui_exception
        self._build()
        self.set_mini_mode(self.settings.mini_mode, persist=False)
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
        menu.add_command(label='模型与截图设置',command=self.show_settings,accelerator='F6')
        self.auto_var=tk.BooleanVar(value=False)
        menu.add_checkbutton(label='自动分析牌面变化',variable=self.auto_var,command=self.toggle_auto)
        view=tk.Menu(menu,tearoff=False)
        self.mini_var=tk.BooleanVar(value=self.settings.mini_mode)
        self.topmost_var=tk.BooleanVar(value=self.settings.mini_topmost)
        view.add_checkbutton(label='迷你模式',variable=self.mini_var,command=lambda:self.set_mini_mode(self.mini_var.get()),accelerator='F8')
        view.add_checkbutton(label='迷你窗口置顶',variable=self.topmost_var,command=self.set_mini_topmost)
        menu.add_cascade(label='视图',menu=view)
        menu.add_separator()
        menu.add_command(label='关于',command=self.show_about)
        menu.add_command(label='使用说明',command=lambda:messagebox.showinfo('使用说明','打开雀魂后点“截图并分析策略”。\n\n默认结构化模式：本地识别并核算事实，模型结合整桌图、分座位点棒图与各家牌河、副露给出攻守建议。先显示初判，再自动补全详解；可以重新截图停止旧分析。三种模式均开启思考，速度取决于画面复杂度和模型服务。\n\nF6 打开设置，Ctrl+S 保存；F8 切换迷你模式。菜单可选自动分析稳定的牌面变化。全程只读取截图。\n\n模型仍可能误判。高亮警告表示核验未通过。',parent=self.root))
        self.root.config(menu=menu)
        self.full_frame=tk.Frame(self.root,bg=self.BG)
        self.full_frame.pack(fill='both',expand=True)
        header=tk.Frame(self.full_frame,bg=self.BG)
        header.pack(fill='x',padx=24,pady=(20,12))
        tk.Label(header,text='雀魂策略助手',font=('Microsoft YaHei UI',23,'bold'),bg=self.BG,fg=self.GOLD).pack(side='left')
        self.badge=tk.Label(header,text='',font=('Microsoft YaHei UI',10),bg=self.BG,fg=self.MUTED)
        self.badge.pack(side='right')
        toolbar=tk.Frame(self.full_frame,bg=self.BG)
        toolbar.pack(fill='x',padx=24,pady=(0,12))
        self.button=ttk.Button(toolbar,text='截图并分析策略',style='Primary.TButton',command=self.capture)
        self.button.pack(side='left')
        self.cancel_button=ttk.Button(toolbar,text='取消分析',command=self.cancel)
        self.status=tk.Label(toolbar,text='准备就绪',bg=self.BG,fg=self.MUTED,font=('Microsoft YaHei UI',10),anchor='w')
        self.status.pack(side='left',padx=18)
        body=tk.PanedWindow(self.full_frame,orient='horizontal',bg=self.BG,sashwidth=8,bd=0)
        body.pack(fill='both',expand=True,padx=24,pady=(0,12))
        left=tk.Frame(body,bg=self.PANEL)
        body.add(left,minsize=220,width=290)
        tk.Label(left,text='本次牌桌',bg=self.PANEL,fg=self.MUTED,font=('Microsoft YaHei UI',11)).pack(anchor='w',padx=16,pady=(16,8))
        self.image_label=tk.Label(left,text='截图后在这里预览\n也可从文件菜单打开图片',bg=self.PANEL,fg=self.MUTED,font=('Microsoft YaHei UI',11),wraplength=250)
        self.image_label.pack(fill='both',expand=True,padx=10,pady=10)
        self.local_label=tk.Label(left,text='观察整张牌桌，结合手牌、牌河、\n副露、宝牌、点棒与巡目分析。',bg=self.PANEL,fg=self.MUTED,font=('Microsoft YaHei UI',10),wraplength=260,justify='left')
        self.local_label.pack(fill='x',padx=16,pady=16)
        right=tk.Frame(body,bg=self.PANEL)
        body.add(right,minsize=460)
        self.title=tk.Label(right,text='等待分析',bg=self.PANEL,fg=self.GOLD,font=('Microsoft YaHei UI',24,'bold'),anchor='w',wraplength=520)
        self.title.pack(fill='x',padx=20,pady=(18,12))
        text_frame=tk.Frame(right,bg=self.PANEL)
        text_frame.pack(fill='both',expand=True,padx=(20,10),pady=(0,12))
        self.text=tk.Text(text_frame,bg=self.PANEL,fg=self.TEXT,insertbackground=self.TEXT,font=('Microsoft YaHei UI',11),wrap='word',relief='flat',borderwidth=0,padx=0,pady=0,spacing1=3,spacing3=5)
        scrollbar=ttk.Scrollbar(text_frame,command=self.text.yview)
        self.text.config(yscrollcommand=scrollbar.set)
        self.text.pack(side='left',fill='both',expand=True);scrollbar.pack(side='right',fill='y')
        self.text.tag_configure('heading',foreground=self.GOLD,font=('Microsoft YaHei UI',12,'bold'),spacing1=15,spacing3=6)
        self.text.tag_configure('warning',foreground=self.RED)
        tk.Label(self.full_frame,text='仅提供可见信息下的策略参考；不读取游戏内存、不操作游戏。游戏画面将发送至所配置的模型服务。',bg=self.BG,fg='#7f8da2',font=('Microsoft YaHei UI',9)).pack(anchor='w',padx=26,pady=(0,12))
        self._build_mini()
        self._set_text([('完整策略分析','点击后读取牌桌并分析策略。\n\n本地算法负责识牌候选与数值核对，策略结论来自启用思考的 Spark 模型。')])
        self.root.bind('<F8>',lambda event:self.set_mini_mode(not self.mini_mode))
        self.root.bind('<Control-comma>',lambda event:self.show_settings())
        self.root.bind('<F6>',lambda event:self.show_settings())
        self._update_badge()

    def _build_mini(self):
        self.mini_frame=tk.Frame(self.root,bg=self.BG,padx=14,pady=10)
        controls=tk.Frame(self.mini_frame,bg=self.BG)
        controls.pack(fill='x',side='top')
        self.mini_button=ttk.Button(controls,text='截图分析',command=self.capture)
        self.mini_button.pack(side='left')
        self.mini_cancel=ttk.Button(controls,text='取消',command=self.cancel,state='disabled')
        self.mini_cancel.pack(side='left',padx=5)
        ttk.Button(controls,text='展开',command=lambda:self.set_mini_mode(False)).pack(side='right')
        ttk.Button(controls,text='设置',command=self.show_settings).pack(side='right',padx=5)
        self.mini_status=tk.Label(self.mini_frame,text='准备就绪',bg=self.BG,fg=self.MUTED,
                                  font=('Microsoft YaHei UI',9),anchor='w',justify='left',wraplength=425)
        self.mini_status.pack(fill='x',pady=(8,2))
        self.mini_title=tk.Label(self.mini_frame,text='等待分析',bg=self.BG,fg=self.GOLD,
                                 font=('Microsoft YaHei UI',22,'bold'),anchor='w',wraplength=425)
        self.mini_title.pack(fill='x',pady=(3,5))
        self.mini_hand=tk.Label(self.mini_frame,text='',bg=self.BG,fg=self.MUTED,
                                font=('Microsoft YaHei UI',9),justify='left',anchor='w',wraplength=425)
        self.mini_hand.pack(fill='x',pady=(0,5))
        content=tk.Frame(self.mini_frame,bg=self.PANEL)
        content.pack(fill='both',expand=True)
        self.mini_text=tk.Text(content,bg=self.PANEL,fg=self.TEXT,font=('Microsoft YaHei UI',10),
                               wrap='word',height=5,relief='flat',padx=8,pady=6)
        scroll=ttk.Scrollbar(content,command=self.mini_text.yview)
        self.mini_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y');self.mini_text.pack(fill='both',expand=True)
        self.mini_frame.bind('<Configure>',lambda event:[label.configure(wraplength=max(280,event.width-30))
                                                        for label in (self.mini_title,self.mini_status,self.mini_hand)])

    def _persist_window_preferences(self):
        self.settings=replace(self.settings,mini_mode=self.mini_mode,mini_topmost=self.topmost_var.get())
        if not self.settings_warning:
            try:
                save_settings(self.settings)
            except Exception as error:
                log_event('window preferences save failed: '+type(error).__name__,'ERROR')
                self._set_status('窗口模式已切换，偏好保存失败',self.RED)

    def set_mini_topmost(self):
        self.root.attributes('-topmost',bool(self.mini_mode and self.topmost_var.get()))
        self._persist_window_preferences()

    def set_mini_mode(self,enabled,persist=True):
        enabled=bool(enabled)
        if enabled != self.mini_mode:
            if enabled:
                if self.root.winfo_width()>100:
                    self.normal_geometry=self.root.geometry()
                self.full_frame.pack_forget()
                self.root.minsize(360,260)
                self.root.geometry('470x390')
                self.mini_frame.pack(fill='both',expand=True)
            else:
                self.mini_frame.pack_forget()
                self.root.minsize(820,700)
                self.root.geometry(self.normal_geometry)
                self.full_frame.pack(fill='both',expand=True)
        self.mini_mode=enabled;self.mini_var.set(enabled)
        self.root.title('雀魂策略助手 v'+APP_VERSION+(' · 迷你' if enabled else ''))
        self.root.attributes('-topmost',bool(enabled and self.topmost_var.get()))
        if persist:
            self._persist_window_preferences()

    def _set_status(self,text,fg=None):
        options={'text':text}
        if fg is not None:options['fg']=fg
        self.status.config(**options)
        self.mini_status.config(**options)

    def _set_heading(self,text,fg):
        self.title.config(text=text,fg=fg)
        self.mini_title.config(text=text,fg=fg)

    def _ui_exception(self,kind,error,trace):
        import traceback
        frames='; '.join('%s:%s' % (frame.name,frame.lineno) for frame in traceback.extract_tb(trace))
        log_event('UI callback '+kind.__name__+' at '+frames,'ERROR')
        messagebox.showerror('界面操作未完成','操作未完成（%s），诊断信息已记录。' % kind.__name__,parent=self.root)

    def _update_badge(self):
        source='雀魂窗口' if self.settings.capture_source=='window' else '全屏'
        current = getattr(self.settings,'analysis_mode','structured')
        mode={'structured':'结构化','jev':'Jev决策','fast':'快速'}.get(current,'完整')
        badge = 'Spark · 快速思考' if current!='full' else 'Spark · '+{'low':'快速思考','medium':'标准思考','high':'深入思考'}[self.settings.reasoning_effort]
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
            self.text.insert('end',humanize_tiles(body)+'\n\n','warning' if heading=='识别核验' else '')
        self.text.config(state='disabled');self.text.yview_moveto(0)
        self.mini_text.config(state='normal');self.mini_text.delete('1.0','end')
        selected=[(h,b) for h,b in sections if h.startswith(('识别核验','当前取舍','进攻路线','防守与风险','后续几巡','原因','正在处理','配置需要检查'))]
        for heading,body in (selected or sections[:1]):
            self.mini_text.insert('end',heading+'\n'+humanize_tiles(body[:260])+'\n\n')
        self.mini_text.config(state='disabled');self.mini_text.yview_moveto(0)

    def _begin(self):
        if self.busy:
            return False
        log_event('analysis started, source=%s' % self.settings.capture_source)
        if self.settings_warning:
            messagebox.showerror('配置需要检查',self.settings_warning,parent=self.root)
            self.show_settings()
            return False
        if not self.settings.api_key or not self.settings.base_url.strip():
            self.show_settings()
            if not self.settings.api_key or not self.settings.base_url.strip():
                return False
        if self.settings.base_url.strip().startswith('http://') and not self.settings.base_url.strip().startswith(('http://127.0.0.1','http://localhost')):
            log_event('WARNING: plaintext http base_url configured; screenshot and key travel unencrypted', 'WARNING')
        self.busy=True;self.generation+=1;self.started=time.perf_counter();self.result=None
        self.initial_advice=None
        self.pipeline.cancelled.clear()
        if self.watcher:self.watcher.reset()
        self.last_phase='正在获取画面'
        if self.image:self.image.close();self.image=None
        self.preview=None;self.image_label.config(image='',text='正在获取新画面…')
        self.button.config(state='disabled');self.cancel_button.pack(side='left',padx=(12,0))
        self.mini_button.config(state='disabled');self.mini_cancel.config(state='normal')
        self.mini_hand.config(text='')
        self._set_heading('正在分析…',self.GOLD)
        self._set_text([('正在处理','正在获取新画面，旧策略已清除。')])
        self._set_status('正在获取画面…',self.MUTED)
        return True

    def capture(self):
        if self.busy:
            self.pending_capture=True
            self.cancel()
            self._set_status('正在停止旧分析，随后抓取新牌面…',self.MUTED)
            return
        if not self._begin():
            return
        generation=self.generation;settings=replace(self.settings)
        if settings.capture_source=='screen':
            self.root.withdraw()
            self.root.after(250,lambda:self._spawn_capture(generation,settings))
        else:
            self._spawn_capture(generation,settings)

    def _hide_for_capture(self, generation):
        """Ask the main thread to withdraw, then wait for its acknowledgement.

        Tk must only be touched on the thread that created it. Posting a
        message lets _poll() hide the window even while a modal dialog or a
        busy handler previously blocked the old root.after(0, ...) path.
        """
        done=threading.Event()
        self.messages.put((generation,'hide',done))
        done.wait(3.0)

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
        from table_viewport import locate_table
        box,source=locate_table(image)
        if box:
            image.info['game_bbox']=box
            image.info['viewport_source']=source
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
                            self._hide_for_capture(generation)
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
        from table_viewport import prepare_table_image
        source=image
        image=prepare_table_image(source)
        source.close()
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
            '模式：快速（默认，只报下一张）／结构化（初判+详解）／完整分析。'
            % APP_VERSION,parent=self.root)

    def _grab_for_watch(self):
        """Screenshot only. Returns a fresh image or None; never touches the game."""
        try:
            if self.closed or self.settings.capture_source!='window':
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
            self._set_status('自动分析已开启：牌面稳定变化后自动给出建议','#8bcdb1')
            log_event('auto analysis on')
        else:
            self.auto_enabled=False
            if self.watcher:
                self.watcher.stop();self.watcher=None
            self._set_status('准备就绪',self.MUTED)
            log_event('auto analysis off')

    def _auto_trigger(self):
        """Called from watcher thread; hop to Tk thread and reuse manual capture."""
        if self.closed or not self.auto_enabled:
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
        self._set_heading(title,self.RED if result.get('validation_warnings') else self.GOLD)
        self._set_text(sections)
        local=result.get('local_evidence',{}).get('hand',{})
        self.local_label.config(text='本地手牌：'+ ' '.join(tile_name(t.get('name') or '?') for t in local.get('tiles',[]))+'\n'+('本地手牌核验通过' if local.get('reliable') else '本地手牌尚待核验'))
        self.mini_hand.config(text=' '.join(tile_name(t.get('name') or '?') for t in local.get('tiles',[])))
        elapsed=result.get('total_seconds',result.get('elapsed_seconds',0))
        stages=result.get('stage_seconds',{})
        stage_names={'local':'识牌','structured':'策略','fast':'策略','jev':'决策','quick':'初判','detail':'详解','vision':'读图','strategy':'策略','pipeline_other':'其他'}
        detail=' · '.join('%s %.1f秒' % (stage_names.get(name,'处理'),value) for name,value in stages.items() if isinstance(value,(int,float)))
        effort = 'low' if result.get('mode') in ('fast','structured') else result.get('reasoning_effort','medium')
        effort = {'low':'快速','medium':'标准','high':'深入'}.get(effort,'标准')
        prefix={'fast':'快速','jev':'Jev决策','structured':'结构化'}.get(result.get('mode'),'完整')
        self._set_status(prefix+' · 分析完成 · %.1f 秒 · 思考 %s%s' % (elapsed,effort,(' · '+detail) if detail else ''),'#8bcdb1')
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
        self.button.config(text='截图并分析策略')
        self.mini_button.config(state='normal');self.mini_cancel.config(state='disabled')
        self.mini_button.config(text='截图分析')
        if self.pending_capture and not self.closed:
            self.pending_capture=False
            self.root.after(100,self.capture)

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
                elif kind=='hide':
                    value.set()  # 过期请求也必须放行等待方
                continue
            if kind=='hide':
                try:
                    self.root.withdraw()
                    # Let the compositor remove the helper before grabbing.
                    self.root.after(180,value.set)
                except Exception:
                    value.set()
            elif kind=='restore':
                self.root.deiconify()
            elif kind=='image':
                self.show_image(value);value.close()
            elif kind=='notice':
                self._set_status(value,self.GOLD)
            elif kind=='progress':
                if self.pipeline.cancelled.is_set():continue
                self.last_phase=PHASE_LABELS.get(value.get('phase'),'正在分析')
                if value.get('phase')=='advice':
                    self.initial_advice=dict(value)
                    self._set_heading(value['title'],self.GOLD)
                    self._set_text([('初步建议',value.get('reason','')),('场面分析','正在补全各家威胁、备选与后续计划，最终结果仍需核验。')])
                    self.button.config(state='normal',text='重新截图')
                    self.mini_button.config(state='normal',text='重新截图')
            elif kind=='result':
                self.show_result(value)
                log_event('analysis completed in %ss' % value.get('total_seconds','?'))
            elif kind=='error':
                log_event('analysis failed: ' + str(value)[:120].replace('\n',' '), 'ERROR')
                if value=='分析已取消' and self.initial_advice and not self.pending_capture:
                    self._set_heading(self.initial_advice['title'],self.GOLD)
                    self._set_text([('本轮初步建议',self.initial_advice.get('reason','')),('详解已停止','这条建议对应刚才的截图；牌面变化后请重新截图。')])
                    self._set_status('已停止后续详解',self.MUTED);self._finish();continue
                self._set_heading('本次分析未完成',self.RED)
                self._set_text([('原因',value),('下一步','可在模型与截图设置中检查连接，或等待牌面稳定后重试。')])
                self._set_status(value[:90],self.RED);self._finish()
        if self.busy:
            self._set_status(self.last_phase+' · 已等待 %d 秒' % (time.perf_counter()-self.started))
        self.root.after(100,self._poll)

    def cancel(self):
        self.pipeline.cancel()
        self._set_status('正在取消…')

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
        try:Path(path).write_text(text,encoding='utf-8')
        except OSError:
            messagebox.showerror('报告未保存','当前目录无法写入，请选择其他保存位置。',parent=self.root)

    def show_settings(self):
        if self.busy:
            messagebox.showinfo('正在分析','请先等待或取消当前分析，再修改模型和截图设置。',parent=self.root);return
        from desktop_widgets import SettingsDialog
        def apply(settings):
            save_settings(settings)
            self.settings=settings;self.settings_warning=''
            self._update_badge();self._set_status('设置已加密保存，重新截图即可使用',self.MUTED);log_event('settings saved')
        dialog=SettingsDialog(self.root,self.settings,apply)
        self.root.wait_window(dialog)

    def close(self):
        self.closed=True;self.generation+=1;self.pipeline.cancel()
        if self.watcher:
            self.watcher.stop();self.watcher=None
        # Drain any queued frames/images so close() is not skipped by the early return.
        self._drain_messages()
        self.pipeline.close()
        if self.image:
            self.image.close()
        self.root.destroy()

    def _drain_messages(self):
        while True:
            try:
                _,kind,value=self.messages.get_nowait()
            except queue.Empty:
                break
            if kind=='image' and hasattr(value,'close'):
                value.close()


def _warm_up_recognition():
    """后台加载识别模板与求解器上下文，避免首次分析多等 2~3 秒。

    只做预热：任何失败都必须静默吞掉，绝不能让预热把应用拖崩。
    """
    try:
        from auto_vision_vendor import reader as vendor_reader
        vendor_reader.load_templates()
    except Exception as error:
        log_event('template warm-up skipped: ' + type(error).__name__, 'WARNING')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image')
    parser.add_argument('--show-report')
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--check-output')
    parser.add_argument('--check-settings-write',action='store_true')
    parser.add_argument('--single-instance-check',action='store_true')
    args=parser.parse_args()
    if args.check_settings_write:
        try:
            save_settings(load_settings())
            return 0
        except (OSError,ValueError):
            return 2
    if args.self_check:
        from PIL import Image
        import cv2
        from desktop_widgets import SettingsDialog
        from table_viewport import prepare_table_image
        from open_efficiency import analyze_open
        settings=load_settings()
        solver=ExactSolver();result=solver.analyze(['1m','2m','3m','4m','5m','6m','7m','8m','9m','2p','3p','4p','5p','4z']);solver.close()
        opened=analyze_open(['1m','2m','3m','4m','5m','6m','7m','8m','9m','1p'],[[{'name':'7z'}]*3])
        text=json.dumps({'ok':result['total']==14 and opened['shanten']==0,'version':APP_VERSION,'widgets_loaded':True,'open_hand_ok':opened['shanten']==0,'model_configured':bool(settings.api_key),'thinking':settings.reasoning_effort,'opencv':cv2.__version__})
        if args.check_output:
            try:Path(args.check_output).write_text(text,encoding='utf-8')
            except OSError:
                log_event('selfcheck passed; report destination is not writable','WARNING')
                if sys.stderr:print('自检完成，但报告目录没有写入权限。',file=sys.stderr)
                return 2
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
    # 模板首次加载约 2.5s（37 组 pattern 图转 float）。放到后台线程预热，
    # 用户第一次点"截图并分析"时模板已在内存里，省掉这段固定等待。
    threading.Thread(target=_warm_up_recognition, daemon=True).start()
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
    try:
        sys.exit(main() or 0)
    except Exception as error:
        log_event('application error: '+type(error).__name__,'ERROR')
        if '--self-check' not in sys.argv:
            messagebox.showerror('助手未能完成操作','操作未完成，请重新打开助手后重试。')
        sys.exit(1)
