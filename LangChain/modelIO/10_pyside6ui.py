import asyncio
import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit,
    QPushButton, QVBoxLayout, QWidget
)
from PySide6.QtCore import QThread, Signal, Qt

# ===================== 你的原有配置 =====================
load_dotenv()
prefix = "DASHSCOPE"
llm = init_chat_model(
    model_provider="openai",
    configurable_fields=["model", "api_key", "base_url"],
    config_prefix=prefix,
    temperature=0.5,
    max_tokens=1024
)
config = {
    "configurable": {
        f"{prefix}_model": os.getenv(f"{prefix}_MODEL"),
        f"{prefix}_api_key": os.getenv(f"{prefix}_API_KEY"),
        f"{prefix}_base_url": os.getenv(f"{prefix}_BASE_URL")
    }
}

# ===================== 异步AI线程封装 =====================
class LlmStreamThread(QThread):
    # 信号：传递分片文本
    chunk_signal = Signal(str)
    # 信号：流式结束
    finish_signal = Signal()

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt

    async def stream_task(self):
        async for chunk in llm.astream(self.prompt, config=config):
            # 发送分片到UI线程
            self.chunk_signal.emit(chunk.content)
        self.finish_signal.emit()

    def run(self):
        # Qt线程内运行asyncio
        asyncio.run(self.stream_task())

# ===================== UI主窗口 =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LangChain PySide6 UI")
        self.resize(800, 600)

        # 组件
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("输入提问，回车或点击发送...")
        self.send_btn = QPushButton("发送")

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.output_text)
        layout.addWidget(self.input_edit)
        layout.addWidget(self.send_btn)

        center_widget = QWidget()
        center_widget.setLayout(layout)
        self.setCentralWidget(center_widget)

        # 绑定事件
        self.send_btn.clicked.connect(self.send_query)
        self.input_edit.returnPressed.connect(self.send_query)

        # 保存线程实例，防止被垃圾回收
        self.stream_thread = None

    def send_query(self):
        prompt = self.input_edit.text().strip()
        if not prompt:
            return
        # 清空输入框，追加用户提问
        self.input_edit.clear()
        self.output_text.append(f"\n【用户】：{prompt}\n【AI】：")

        # 创建并启动流式线程
        self.stream_thread = LlmStreamThread(prompt)
        self.stream_thread.chunk_signal.connect(self.append_stream_text)
        self.stream_thread.finish_signal.connect(self.stream_finish)
        self.stream_thread.start()

    def append_stream_text(self, text: str):
        # 实时追加流式文本，光标自动移到末尾
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.output_text.setTextCursor(cursor)
        self.output_text.ensureCursorVisible()

    def stream_finish(self):
        self.output_text.append("\n——————回答结束——————\n")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())