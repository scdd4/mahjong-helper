"""Responsive settings form with a save bar outside the scrolling content."""
from dataclasses import replace
import tkinter as tk
from tkinter import messagebox, ttk
from desktop_runtime import log_event


MODE_LABELS = {'fast': '快速（只报下一张，推荐）', 'structured': '结构化（初判+详解）', 'full': '完整分析'}


def settings_from_form(initial, values):
    """Shared validation for the settings form; retains window preferences."""
    reverse = {label: mode for mode, label in MODE_LABELS.items()}
    if values['analysis_mode'] not in reverse:
        raise ValueError('分析模式无效，请选择：'+ '、'.join(MODE_LABELS.values()))
    return replace(initial, base_url=values['base_url'].strip(), model=values['model'].strip(),
                   api_key=values['api_key'].strip(), reasoning_effort={'快速':'low','标准':'medium','深入':'high'}.get(values['reasoning_effort'],values['reasoning_effort']),
                   max_tokens=int(values['max_tokens']), timeout=int(values['timeout']),
                   capture_source='window' if values['capture_source'] == '雀魂窗口' else 'screen',
                   analysis_mode=reverse[values['analysis_mode']],
                   # 用户在这里保存过就算主动选择，之后不再被默认值迁移覆盖。
                   mode_explicit=True).validate()


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, settings, on_save):
        super().__init__(parent)
        self.initial, self.on_save = settings, on_save
        self.title('模型与截图设置')
        self.transient(parent)
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        width, height = min(680, screen_w - 40), min(760, screen_h - 100)
        self.geometry('%dx%d+%d+%d' % (width, height,
                      max(0, min(parent.winfo_rootx() + 30, screen_w - width)),
                      max(0, min(parent.winfo_rooty() + 30, screen_h - height - 60))))
        self.minsize(min(480, width), min(350, height))
        self.resizable(True, True)
        self.values = {}

        # The footer is allocated before the body and never scrolls away.
        self.footer = ttk.Frame(self, padding=(16, 10))
        self.footer.pack(side='bottom', fill='x')
        ttk.Label(self.footer, text='Ctrl+S 保存设置').pack(side='left')
        ttk.Button(self.footer, text='取消', command=self.destroy).pack(side='right', padx=(8, 0))
        self.save_button = ttk.Button(self.footer, text='保存设置', command=self.save)
        self.save_button.pack(side='right')
        ttk.Separator(self, orient='horizontal').pack(side='bottom', fill='x')

        body = ttk.Frame(self)
        body.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(body, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(body, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.form = ttk.Frame(self.canvas, padding=20)
        window = self.canvas.create_window((0, 0), window=self.form, anchor='nw')
        self.form.bind('<Configure>', lambda event: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda event: self.canvas.itemconfigure(window, width=event.width))
        self.bind('<MouseWheel>', self._scroll)
        self.bind('<Button-4>', lambda event: self.canvas.yview_scroll(-3, 'units'))
        self.bind('<Button-5>', lambda event: self.canvas.yview_scroll(3, 'units'))

        for index, (field, label) in enumerate((('base_url', 'Base URL'), ('model', '模型名称'), ('api_key', 'API 密钥'))):
            ttk.Label(self.form, text=label).grid(row=index*2, column=0, columnspan=2, sticky='w', pady=(4, 4))
            value = tk.StringVar(value=getattr(settings, field))
            self.values[field] = value
            entry = ttk.Entry(self.form, textvariable=value, show='•' if field == 'api_key' else '')
            entry.grid(row=index*2+1, column=0, columnspan=2, sticky='ew', pady=(0, 12))

        options = [
            ('reasoning_effort', '完整模式思考强度', ('快速', '标准', '深入'), {'low':'快速','medium':'标准','high':'深入'}[settings.reasoning_effort]),
            ('max_tokens', '完整模式输出上限', ('8192', '16384', '32768'), str(settings.max_tokens)),
            ('timeout', '等待时间（秒）', ('600', '900', '1200'), str(settings.timeout)),
            ('capture_source', '截图来源', ('雀魂窗口', '全屏'), '雀魂窗口' if settings.capture_source == 'window' else '全屏'),
            ('analysis_mode', '分析模式', tuple(MODE_LABELS.values()), MODE_LABELS[settings.analysis_mode]),
        ]
        for row, (field, label, choices, initial) in enumerate(options, 6):
            ttk.Label(self.form, text=label).grid(row=row, column=0, sticky='w', pady=8)
            value = tk.StringVar(value=initial)
            self.values[field] = value
            ttk.Combobox(self.form, textvariable=value, values=choices, state='readonly', width=20).grid(row=row, column=1, sticky='ew', padx=(12, 0))
        description = ('快速模式（默认）只回答这一张打什么；不是你回合时回答轮到你先考虑哪些牌，约 5 秒。'
                       '结构化模式先给初判再补全场面详解，完整模式展开攻守与后续计划，两者耗时较长。'
                       '\n三种模式均开启思考。结构化和快速模式使用快速思考档位。'
                       '\n\n密钥使用当前 Windows 用户加密保存。识别到的游戏画面会发送到配置的模型服务。')
        help_label = ttk.Label(self.form, text=description, wraplength=560, justify='left')
        help_label.grid(row=11, column=0, columnspan=2, sticky='ew', pady=(14, 6))
        self.form.bind('<Configure>', lambda event: (self.canvas.configure(scrollregion=self.canvas.bbox('all')),
                                                    help_label.configure(wraplength=max(240, event.width - 45))))
        self.form.columnconfigure(0, weight=1)
        self.form.columnconfigure(1, weight=1)
        self.bind('<Control-s>', self.save)
        self.bind('<Escape>', lambda event: self.destroy())
        self.grab_set()
        self.after(200,self._record_layout)

    def _record_layout(self):
        if not self.winfo_exists():return
        top=self.save_button.winfo_rooty()
        inside=(self.winfo_rooty()<=top and top+self.save_button.winfo_height()<=self.winfo_rooty()+self.winfo_height())
        log_event('settings save footer within window: '+str(inside))

    def _scroll(self, event):
        if event.delta:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units')
        return 'break'

    def save(self, event=None):
        try:
            value = settings_from_form(self.initial, {key: variable.get() for key, variable in self.values.items()})
            self.on_save(value)
        except Exception as error:
            messagebox.showerror('设置未保存', str(error), parent=self)
            return 'break'
        self.destroy()
        return 'break'
