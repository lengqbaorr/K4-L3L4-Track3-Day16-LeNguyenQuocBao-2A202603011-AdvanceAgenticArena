"""LỚP `budget_policy` — bài giảng Day 16, §3 (Budgets & Control Flow).

NHIỆM VỤ: kế hoạch của mô hình dài đúng 11 lượt gọi công cụ, bất kể brief
cho ngân sách bao nhiêu — và BỐN lượt cuối là rác có chủ ý: một lần search
lặp lại, một phép tính vô nghĩa, hai lần fetch lại tài liệu đã có trong
tay. Phần việc hữu ích nằm ở ĐẦU kế hoạch, nên cắt phần đuôi không mất một
điểm grounding nào mà lấy trọn phần điểm efficiency về tool call và token.

TÍN HIỆU:

    ctx.tools.calls >= ctx.max_tool_calls - reserve

CÁCH DỪNG: thêm `FINALIZE_SENTINEL` vào bên trong MỘT CÂU tiếng Việt bình
thường và đẩy vào cuối danh sách message trong `before_model`. `MockModel`
khoá theo token; một mô hình thật thì nghe câu tiếng Việt bao quanh nó.
Viết như vậy để cùng một lớp chạy được trên cả hai đường.

SENTINEL KHÔNG PHẢI TUỲ CHỌN — và không chỉ vì chuyện dừng.
`arena.model._first_user_content` lấy message user CUỐI CÙNG trước lượt
assistant đầu tiên làm câu hỏi của brief, và nó bỏ qua đúng những message
có mang `FINALIZE_SENTINEL`. Nếu bạn chèn một câu nhắc trơn không có
sentinel, mô hình sẽ đi search CHÍNH CÂU NHẮC ĐÓ: mọi brief truy xuất
cùng một mớ tài liệu vô can, mọi bậc thang điểm dịch chuyển đúng 0.00, và
không có một dòng lỗi nào báo cho bạn biết.

TRẢ VỀ `messages + [...]`, ĐỪNG `messages.append(...)`. Agent áp dụng
`before_model` lên một BẢN SAO của lịch sử, nên trả về danh sách mới nghĩa
là "nhắc trong đúng lượt này"; append vào chính danh sách được truyền vào
thì lời nhắc dính vĩnh viễn.

`reserve` KHÔNG PHẢI TRANG TRÍ: `Tools.calls` ĐẾM CẢ `submit`, và scorer
cũng đếm như vậy. Brief cho `max_tool_calls: 8` nghĩa là bảy lượt hữu ích
cộng một lượt submit. Dừng ở `calls >= 8` là tiêu lố đúng một lượt, lần
nào cũng lố.

MỘT HOOK LÀ CHƯA ĐỦ — ĐÃ ĐO. `before_model` chỉ chặn được khi mỗi lượt
model tiêu đúng MỘT lượt công cụ. Không phải vậy: lớp `retry` (§7) có thể
tiêu ba lượt trong CÙNG một vòng, nên một vòng bắt đầu khi còn thiếu đúng
một lượt vẫn kết thúc ở trên ngưỡng. Đo trên full stack: 34/120 lượt chạy
kết thúc ở 9+ lượt gọi trong khi brief cho 8, efficiency 12.06 thay vì
14.24 — trong khi `budget_policy` chạy MỘT MÌNH thì sạch cả 120 lượt.
Vì thế lớp này có thêm `wrap_tool_call`: khi ngân sách chỉ còn phần dự
trữ, TỪ CHỐI gọi công cụ (trả về `ToolResult(ok=False, ...)`, đừng raise —
agent phải sống để còn chốt FINAL). Nửa còn lại nằm ở `retry`: hook
`wrap_tool_call` của `budget_policy` bọc NGOÀI vòng lặp thử lại nên không
nhìn thấy các lượt gọi lại; chỉ chính `retry` mới chặn được `retry`.

CẢNH BÁO ĐÃ ĐO ĐƯỢC — ĐỪNG NÉN NGỮ CẢNH Ở ĐÂY. `before_model` trông rất
hợp lý để "tóm tắt cho gọn", nhưng `MockModel` chỉ trích được câu nào
xuất hiện NGUYÊN VĂN trong danh sách message NÓ ĐANG NHẬN. Một lớp nén
ngữ cảnh tử tế làm mô hình mất khả năng trích dẫn chính những tài liệu nó
vừa đọc: -47.16 điểm trên full stack (92.52 -> 45.36), không có một
thông báo lỗi nào.

CÔNG CỤ CÓ SẴN:
    from arena.model import FINALIZE_SENTINEL
    from arena.tools import ToolResult
    ctx.tools.calls      -> số lượt gọi công cụ đã dùng (kể cả submit)
    ctx.max_tool_calls   -> ngân sách của brief, hoặc None nếu brief không đặt

Cài đặt:  ReActAgent(..., middleware=[..., BudgetPolicy(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import time

from arena.model import FINALIZE_SENTINEL, MockModel
from arena.tools import ToolResult

from harness.middleware import Middleware

#: Dành lại cho lượt `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1

NUDGE = (
    "Ngân sách công cụ đã hết. Hãy trả lời ngay bằng bằng chứng đang có, "
    f"không gọi thêm công cụ nào nữa. {FINALIZE_SENTINEL}"
)


class BudgetPolicy(Middleware):
    """Ép mô hình chốt FINAL ngay khi ngân sách công cụ đã tiêu hết."""

    name = "budget_policy"

    def __init__(self, reserve: int = DEFAULT_RESERVE) -> None:
        self.reserve = max(0, int(reserve))

    def before_agent(self, ctx) -> None:
        ctx.state["budget_started_at"] = time.monotonic()

    def _spent(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        if limit is not None and ctx.tools.calls >= limit - self.reserve:
            return True
        max_tokens = ctx.budget.get("max_tokens")
        if max_tokens is not None:
            used = ctx.state.get("budget_tokens_used", 0)
            last_turn_cost = ctx.state.get("budget_last_turn_cost", 0)
            if used + last_turn_cost > max_tokens:
                return True
        max_seconds = ctx.budget.get("max_seconds")
        started = ctx.state.get("budget_started_at")
        return (max_seconds is not None and started is not None
                and time.monotonic() - started > 0.8 * max_seconds)

    def before_model(self, ctx, messages):
        if self._spent(ctx):
            return messages + [{"role": "user", "content": NUDGE}]
        if any(message.get("role") == "assistant" for message in messages):
            question = ""
            for message in messages:
                if message.get("role") == "assistant":
                    break
                if message.get("role") in ("user", "human"):
                    content = message.get("content", "")
                    content = content if isinstance(content, str) else str(content)
                    if FINALIZE_SENTINEL not in content:
                        question = content
            hint = (
                "Kết quả search là trích đoạn ngắn: fetch_doc ngay 1–3 tài liệu liên quan; không search quá hai lần liên tiếp nếu chưa fetch_doc. "
                "Kho có các loại văn bản: Văn bản chính thức, Hỏi & Đáp, Báo cáo, Ghi chú. "
                "Hỏi về con số/thống kê/số vụ: search '<tên chủ đề> báo cáo' và fetch tài liệu Báo cáo; "
                "hỏi về quy định: search '<tên chủ đề> văn bản chính thức'. "
                "Lấy tên chủ đề từ tiêu đề kết quả search (phần trước dấu —). "
                "Claim chép nguyên dòng, không cắt, thêm hay sửa."
            )
            if all(label in question.lower() for label in ("(a)", "(b)", "(c)")):
                hint += (
                    " Chỉ khi câu hỏi liệt kê lựa chọn dạng (a), (b), (c), hãy ghi đúng một lựa chọn "
                    "đã chọn, sao chép nguyên văn, trong khóa 'verdict' của FINAL."
                )
            return messages + [{"role": "user", "content": hint}]
        return messages

    def wrap_tool_call(self, ctx, call, name, args):
        if self._spent(ctx):
            return ToolResult(ok=False, content="", error="Hết ngân sách công cụ")
        model = getattr(ctx.model, "inner", ctx.model)
        if name == "search" and not isinstance(model, MockModel):
            search_args = dict(args) if isinstance(args, dict) else {}
            given_k = search_args.get("k") or 5
            if isinstance(given_k, bool) or not isinstance(given_k, int):
                given_k = 5
            search_args["k"] = max(given_k, 10)
            return call(name, search_args)
        return call(name, args)

    def wrap_model_call(self, ctx, call, messages):
        response = call(messages)
        state = ctx.state
        state["budget_tokens_used"] = (
            state.get("budget_tokens_used", 0)
            + response.prompt_tokens
            + response.completion_tokens
        )
        state["budget_last_turn_cost"] = response.prompt_tokens + response.completion_tokens
        return response
