import asyncio
import os
import sys
import base64
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap

# ===================== 工具函数 =====================
def file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    base64_str = base64.b64encode(bytes_data).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"

# ===================== LLM配置 =====================
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

prompt_template = ChatPromptTemplate([
    {"role": "system", "content": "你是专业的多模态内容分析助手。"},
    {"role": "user", "content": [
        {"type": "text", "text": "用中文简短描述图片内容"},
        {"type": "image_url", "image_url": {"url": "{image_url}"}}
    ]}
])

# ===================== AI线程 =====================
class LlmStreamThread(QThread):
    chunk_signal = Signal(str)
    finish_signal = Signal()
    stop_flag = False

    def __init__(self, prompt_msg):
        super().__init__()
        self.prompt_msg = prompt_msg
        self.stop_flag = False

    async def stream_task(self):
        async for chunk in llm.astream(self.prompt_msg, config=config):
            if self.stop_flag:
                break
            self.chunk_signal.emit(chunk.content)
        self.finish_signal.emit()

    def run(self):
        self.stop_flag = False
        asyncio.run(self.stream_task())

    def stop_generate(self):
        self.stop_flag = True

# ===================== 纯代码UI窗口（不需要chat.ui文件） =====================
class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多模态对话")
        self.resize(850, 650)

        # 构建界面控件
        central = QWidget()
        layout = QVBoxLayout(central)

        self.btn_upload_img = QPushButton("上传图片")
        self.lbl_img_preview = QLabel()
        self.lbl_img_preview.setAlignment(Qt.AlignCenter)
        self.lbl_img_preview.setMaximumHeight(180)

        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)

        self.edt_input = QLineEdit()
        self.edt_input.setPlaceholderText("输入提问，回车发送")

        row_btn = QWidget()
        row_layout = QVBoxLayout(row_btn)
        self.btn_send = QPushButton("发送")
        self.btn_stop = QPushButton("停止生成")
        row_layout.addWidget(self.btn_send)
        row_layout.addWidget(self.btn_stop)

        # 布局挂载
        layout.addWidget(self.btn_upload_img)
        layout.addWidget(self.lbl_img_preview)
        layout.addWidget(self.txt_output)
        layout.addWidget(self.edt_input)
        layout.addWidget(row_btn)

        self.setCentralWidget(central)

        # 变量缓存
        self.current_image_b64 = None
        self.stream_thread = None

        # 绑定信号
        self.btn_upload_img.clicked.connect(self.select_image)
        self.btn_send.clicked.connect(self.send_question)
        self.btn_stop.clicked.connect(self.stop_ai)
        self.edt_input.returnPressed.connect(self.send_question)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", os.path.expanduser("~"),
            "图片 (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        pix = QPixmap(file_path)
        pix = pix.scaled(self.lbl_img_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_img_preview.setPixmap(pix)
        self.current_image_b64 = file_to_base64(file_path)
        self.txt_output.append(f"\n已加载图片：{os.path.basename(file_path)}")

    def send_question(self):
        prompt_text = self.edt_input.text().strip()
        self.edt_input.clear()

        if self.current_image_b64:
            msg = prompt_template.invoke({"image_url": self.current_image_b64})
            self.txt_output.append(f"\n【用户】图提问：{prompt_text if prompt_text else '描述图片'}\n【AI】")
            self.current_image_b64 = None
            self.lbl_img_preview.clear()
        else:
            if not prompt_text:
                return
            msg = prompt_text
            self.txt_output.append(f"\n【用户】{prompt_text}\n【AI】")

        self.stream_thread = LlmStreamThread(msg)
        self.stream_thread.chunk_signal.connect(self.append_text)
        self.stream_thread.finish_signal.connect(self.on_stream_end)
        self.stream_thread.start()

    def append_text(self, text: str):
        cursor = self.txt_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.txt_output.setTextCursor(cursor)
        self.txt_output.ensureCursorVisible()

    def on_stream_end(self):
        self.txt_output.append("\n----------回答完成----------\n")
        self.stream_thread = None

    def stop_ai(self):
        if self.stream_thread and self.stream_thread.isRunning():
            self.stream_thread.stop_generate()

# ===================== 入口 =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ChatWindow()
    win.show()
    sys.exit(app.exec())