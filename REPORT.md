# Báo cáo Day 17 – Memory Systems for AI Agent

**Sinh viên:** Nguyễn Hải Nam – 2A202602476

## 1. Kiến trúc memory

| Lớp | Baseline | Advanced | Cài đặt |
|---|---|---|---|
| Short-term (trong thread) | Toàn bộ message của thread, đọc lại mỗi lượt | Message gần nhất (`keep_messages=4`) | `SessionState` / `CompactMemoryManager` |
| Persistent (qua thread/phiên) | ❌ không có | `state/profiles/<user>/User.md` | `UserProfileStore` |
| Compact (nén hội thoại dài) | ❌ | Vượt `800` token → message cũ gộp vào summary có giới hạn 8 dòng | `CompactMemoryManager._compact()` |

Luồng xử lý một lượt của Advanced (`src/agent_advanced.py`):

```
message → extract_profile_updates()   (fact có cấu trúc + confidence ≥ 0.6)
        → upsert_fact() vào User.md   (fact mới ghi đè fact cũ)
        → CompactMemoryManager.append (tự compact khi vượt ngưỡng)
        → prompt = User.md + summary + recent messages
        → trả lời → cập nhật bộ đếm token
```

`User.md` cuối bộ Standard (355 bytes):

```markdown
# User Profile: dungct

- name: DũngCT
- location: Huế
- profession: MLOps engineer
- interests: Python, AI ứng dụng, AI agent
- favorite_drink: cà phê sữa đá
- response_style: ngắn gọn, có ví dụ thực tế, bullet, có ví dụ thực chiến, ưu tiên trade-off, có cấu trúc
- favorite_food: mì Quảng
- pet: corgi tên Bơ
```

Cả hai agent đều có **chế độ offline deterministic** (mặc định trong benchmark/test, không cần API key) và **chế độ live** bằng LangChain `create_agent` + `InMemorySaver`. Ở chế độ live, Advanced có thêm tool đọc/ghi/sửa `User.md`, `dynamic_prompt` chèn `User.md` vào prompt và `SummarizationMiddleware`. Hỗ trợ 6 provider: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`.

## 2. Kết quả benchmark

Lệnh: `python src/benchmark.py` (offline, `compact_threshold_tokens=800`, `compact_keep_messages=4`). Kết quả lặp lại y hệt giữa các lần chạy.

**Standard Benchmark** – `data/conversations.json` (10 hội thoại, 101 lượt, 14 câu hỏi recall)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------:|------:|------:|------:|----:|---:|
| Baseline | 2066 | 10843 | 0.000 | 0.000 |   0 | 0 |
| Advanced | 2337 | 19312 | 1.000 | 1.000 | 355 | 0 |

**Long-Context Stress Benchmark** – `data/advanced_long_context.json` (1 hội thoại, 16 lượt dài, 3 câu hỏi recall)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------:|------:|------:|------:|----:|---:|
| Baseline | 2502 | 21563 | 0.000 | 0.000 |   0 | 0 |
| Advanced | 2649 |  9980 | 1.000 | 1.000 | 259 | 6 |

**Guardrail Benchmark (bonus)** – `data/guardrail_cases.json` (3 hội thoại, 19 lượt, 4 câu hỏi recall). Đây là bộ dữ liệu do mình tự viết. Mỗi hội thoại nhắm vào một bẫy: câu phỏng đoán hoặc câu điều kiện về nơi ở, phủ định không được nhắc lại ở cuối, và correction giá trị yêu thích. Mỗi câu hỏi có thêm trường `expected_not_contains` liệt kê các fact cũ/sai.

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------:|------:|------:|------:|----:|---:|
| Baseline | 349 | 1168 | 0.000 | 0.000 |   0 | 0 |
| Advanced | 421 | 2081 | 1.000 | 1.000 | 180 | 0 |

Cách đo:
- **Agent tokens only**: tổng token của message người dùng và câu trả lời trong các thread hội thoại (ước lượng ~4 ký tự/token).
- **Prompt tokens processed**: tổng ngữ cảnh đưa vào ở từng lượt. Baseline tính toàn bộ lịch sử thread; Advanced tính `User.md` + summary + message gần nhất.
- **Cross-session recall**: câu hỏi recall được hỏi ở **thread mới**. Điểm: 1 nếu đủ mọi chuỗi `expected_contains`, 0.5 nếu đủ một phần, 0 nếu không có. Câu trả lời vẫn nêu một fact trong `expected_not_contains` bị **0 điểm**, vì giữ song song fact cũ và fact mới là sai.
- **Response quality**: ở chế độ offline dùng heuristic thang 0–1, `coverage × (0.7 + 0.15·ngắn gọn + 0.15·có bullet)`; coverage bị chia đôi nếu câu trả lời thừa nhận "chưa có thông tin" và chia đôi thêm lần nữa nếu có fact cũ. Ở chế độ `--live` dùng **LLM-as-judge** (`judge_model` trong config) chấm 0–10: 6 điểm cho fact đúng, 2 điểm cho việc không có fact cũ, 2 điểm cho sự ngắn gọn. Nếu judge lỗi thì quay về heuristic và in số lần phải quay về.
- Token ở thread recall không được cộng vào hai cột token. Quy tắc này áp dụng giống nhau cho cả hai agent.

## 3. Phân tích

### 3.1 Vì sao Advanced recall tốt hơn
Câu hỏi recall được hỏi ở thread mới. Baseline có thread trống nên chỉ trả lời "chưa có thông tin" → recall 0. Đây là hành vi đúng: baseline không được giả vờ có long-term memory. Test `test_baseline_remembers_within_same_thread` xác nhận baseline vẫn nhớ trong cùng thread. Advanced đọc `User.md` nên trả lời đúng cả 17 câu, kể cả những câu có correction:

- Standard: nơi ở **Đà Nẵng → Huế** (conv-03), nghề **backend → MLOps engineer** (conv-06). Câu "nhắc lại Đà Nẵng như ví dụ cũ" ở conv-10 không làm đổi fact.
- Stress: nơi ở **Huế → Đà Nẵng**. Nhiễu "Hà Nội chỉ là nơi đi họp" và "product manager chỉ là câu đùa" không được ghi vào `User.md`.

### 3.2 Vì sao Advanced tốn hơn ở hội thoại ngắn
Ở bộ Standard, Advanced tốn **+78% prompt tokens** (19 312 so với 10 843) và **+13% agent tokens**. Lý do:
- Mỗi lượt Advanced đều kéo theo `User.md` (~80 token). Hội thoại ngắn (~10 lượt × ~20 token) nên đây là chi phí cố định lớn so với chính lịch sử hội thoại.
- Hội thoại chưa bao giờ vượt ngưỡng 800 token nên **không có lần compact nào**: chỉ có chi phí mà chưa có lợi ích.
- Câu trả lời xác nhận "Đã ghi nhớ vào User.md: …" dài hơn câu "Mình đã ghi nhận." của baseline.

Đổi lại, recall tăng từ 0 lên 1.0. Với hội thoại ngắn, đây là một **trade-off**: trả token để có trí nhớ dài hạn, chứ không phải tối ưu chi phí.

### 3.3 Vì sao compact thắng ở hội thoại dài, và chủ yếu ở prompt tokens
Ở bộ Stress, Advanced giảm **53,7% prompt tokens** (9 980 so với 21 563) sau 6 lần compact.

- Baseline phải đọc lại toàn bộ lịch sử ở mỗi lượt. Với n lượt, chi phí là `Σ (1..n)` → tăng theo **bậc 2** theo số lượt.
- Advanced giữ ngữ cảnh mỗi lượt quanh một mức trần (≈ `User.md` + summary ≤ 8 dòng + 4 message) → chi phí tăng **gần tuyến tính**.
- `Agent tokens only` **không giảm** (2 649 so với 2 502), vì compact không thay đổi lượng văn bản người dùng gửi và agent sinh ra. Nó chỉ cắt phần ngữ cảnh cũ bị kéo theo. Vì vậy compact tối ưu chủ yếu ở `Prompt tokens processed`.

Ablation trên bộ Stress (các cấu hình khác giữ nguyên; chạy lại bằng `python src/ablation.py`):

| Cấu hình Advanced | Prompt tokens | Compactions | Recall |
|---|---:|---:|---:|
| Không compact (ngưỡng = ∞) | 23 449 | 0 | 1.0 |
| ngưỡng 2000 | 16 993 | 1 | 1.0 |
| ngưỡng 1200 | 13 510 | 3 | 1.0 |
| **ngưỡng 800 (mặc định)** | **9 980** | **6** | 1.0 |
| ngưỡng 500 | 8 319 | 24 | 1.0 |
| ngưỡng 800, keep = 8 | 12 310 | 21 | 1.0 |
| *Baseline* | *21 563* | *0* | *0.0* |

Nhận xét:
- **Không compact thì Advanced còn đắt hơn Baseline** (23 449 so với 21 563, do overhead `User.md`). Lợi thế ở hội thoại dài đến từ compact, chứ không đến từ `User.md`.
- Ngưỡng càng thấp thì càng tiết kiệm nhưng lợi ích giảm dần (800 → 500 chỉ bớt thêm ~17%), trong khi số lần compact tăng gấp 4. Với summary dùng LLM thật, mỗi lần compact là **một lần gọi model**, nên compact quá dày sẽ tốn chi phí ở chỗ khác.
- `keep = 8` gây **thrashing**: 8 message giữ lại đã gần chạm ngưỡng, nên lượt nào cũng phải compact (21 lần). Ràng buộc thiết kế: tổng token của các message giữ lại phải nhỏ hơn rõ rệt so với `threshold`.
- Recall vẫn là 1.0 ở mọi ngưỡng, vì fact quan trọng đã nằm ở `User.md` chứ không phụ thuộc vào summary. Đây chính là lý do cần **tách persistent memory khỏi compact memory**.

### 3.4 Memory file tăng trưởng và rủi ro
- `User.md` tăng 355 bytes sau 10 phiên (Standard) và 259 bytes sau 16 lượt dài (Stress). Nhờ `upsert_fact` **ghi đè theo key** và bỏ qua khi giá trị không đổi (`test_upsert_is_idempotent`), kích thước file bị giới hạn theo số field chứ không theo số lượt hội thoại.
- **Rủi ro phình to**: các field dạng danh sách (`response_style`, `interests`) được gộp theo kiểu hợp tập hợp nên vẫn tăng dần: `response_style` hiện đã có 6 tag, trong đó "có ví dụ thực tế" và "có ví dụ thực chiến" gần trùng nghĩa. Về lâu dài cần chuẩn hoá đồng nghĩa và giới hạn số tag hoặc dùng memory decay. Mọi lượt đều đọc `User.md`, nên mỗi byte tăng thêm là chi phí bị nhân lên theo số lượt.
- **Rủi ro lưu sai fact**: extractor dựa trên regex và whitelist (địa danh, nghề nghiệp), nên câu viết khác mẫu sẽ bị bỏ sót (false negative), còn câu bẫy mới có thể bị ghi sai (false positive). Một fact sai trong `User.md` sẽ **lan sang mọi phiên sau**: nguy hiểm hơn lỗi trong short-term memory.
- **Rủi ro summary mất thông tin**: summary heuristic chỉ giữ ~120 ký tự đầu của 6 message người dùng gần nhất, nên chi tiết tin tức ở đầu chuỗi stress bị mất. Nó vẫn chấp nhận được vì fact ổn định đã được tách sang `User.md`, nhưng câu hỏi về nội dung cũ trong cùng thread sẽ trả lời kém.
- **Quyền riêng tư**: `User.md` là dữ liệu cá nhân dạng văn bản thuần, cần chính sách xoá/xem lại cho người dùng (đã nằm trong `.gitignore`).

## 4. Bonus

Ablation guardrail của Advanced (`python src/ablation.py`). "Answers with stale fact" là số câu trả lời recall vẫn nêu fact trong `expected_not_contains` (chỉ bộ Guardrail có trường này):

| Advanced variant | Recall – Standard | Recall – Stress | Recall – Guardrail | Answers with stale fact |
|---|---:|---:|---:|---:|
| **full system** | **1.000** | **1.000** | **1.000** | **0** |
| không có confidence threshold | 1.000 | 1.000 | 0.750 | 1 |
| không xử lý phủ định | 0.929 | 1.000 | 0.750 | 1 |
| tắt cả hai | 0.929 | 1.000 | 0.500 | 2 |

Hai guardrail **bổ trợ nhau**: mỗi cái chặn một loại lỗi khác nhau, tắt cái nào cũng mất một câu, tắt cả hai thì recall Guardrail giảm một nửa. Bộ Stress luôn đạt 1.0 vì dữ liệu có câu nhắc lại fact đúng ở cuối, và chính điều đó là lý do mình phải tự viết bộ Guardrail để lộ ra lỗi.

### 4.1 Conflict handling: correction ghi đè, không giữ song song fact cũ
- **Vấn đề:** người dùng đính chính (Đà Nẵng → Huế, backend → MLOps); nếu append tự do, `User.md` sẽ giữ cả hai giá trị và agent có thể trả lời bằng fact cũ.
- **Giải pháp:**
  1. `upsert_fact` ghi đè theo key, nên mỗi field chỉ có đúng một dòng.
  2. Trong extractor, xét **phủ định theo từng mệnh đề**: câu được tách theo `, ; :` / "chứ" / "nhưng"; một thực thể bị huỷ nếu phần đứng trước nó trong cùng mệnh đề chứa "không còn", "lúc đầu", "trước đó", "ví dụ cũ", "đùa", "hay là"…, hoặc ngay sau nó là "chỉ là".
  3. Thực thể chỉ được nhận khi có **trigger khẳng định** ("đang ở", "làm việc ở", "chuyển sang", "nghề"…).
  4. Giá trị yêu thích dừng ở từ đối lập: "đồ uống yêu thích là matcha latte **chứ không phải** trà đào" → `matcha latte`.
- **Hiệu quả (ablation):** tắt xử lý phủ định thì recall Standard giảm **1.000 → 0.929**. Câu "mình làm MLOps engineer **chứ không còn là backend engineer**" (conv-09) khiến nghề bị ghi ngược thành backend engineer, nên conv-09 và conv-10 trả lời sai. Trên bộ Guardrail, recall giảm **1.000 → 0.750**: câu "**Lúc trước** mình làm data engineer" ghi đè `ML engineer`. Test: `test_correction_overwrites_stale_fact`, `test_noise_does_not_override_profile`, `test_correction_value_stops_at_contrast_word`, `test_guardrails_are_needed_on_guardrail_dataset`.
- **Rủi ro:** fact mới luôn thắng, nên một câu khẳng định sai (hoặc câu đùa không có từ "đùa") sẽ ghi đè fact đúng. Hệ thống không lưu lịch sử nên không rollback được.

### 4.2 Confidence threshold
- **Vấn đề:** không phải câu nào khớp pattern cũng là fact: có câu hỏi, câu điều kiện, câu phỏng đoán.
- **Giải pháp:** `extract_profile_facts()` trả về `ProfileFact(key, value, confidence, evidence)`. Điểm gốc theo độ chắc của pattern ("tên là", "món ăn yêu thích là": 0.95; dựa trên trigger: 0.85; "uống …": 0.75). Câu có hedge ("có lẽ", "hình như", "giả sử", "đùa") bị trừ 0.4, câu điều kiện "nếu" bị trừ 0.3 (riêng style trả lời được miễn, vì chỉ dẫn thường có dạng "nếu bạn giải thích thì…"). Chỉ fact có `confidence ≥ 0.6` mới được ghi. Câu hỏi bị loại từ đầu.
- **Hiệu quả:** trên bộ Standard và Stress, bỏ ngưỡng không làm đổi kết quả, vì hai bộ này không có câu phỏng đoán hay câu điều kiện về fact. Trên bộ Guardrail, bỏ ngưỡng làm recall giảm **1.000 → 0.750**: câu "Có lẽ tháng sau mình sẽ chuyển ra Hà Nội" (0.45) và "Nếu công ty mở chi nhánh thì mình sẽ chuyển về Đà Nẵng" (0.55) ghi đè nơi ở Huế, và agent trả lời "Đà Nẵng". Test: `test_confidence_threshold_skips_hedged_facts`, `test_guardrails_are_needed_on_guardrail_dataset`.
- **Rủi ro:** có false negative thật. Câu "mình đang ở Huế để dùng ví dụ địa phương **nếu** cần" (conv-08) bị loại (0.55) chỉ vì chữ "nếu" nằm ở mệnh đề phụ. Ở đây không gây hại vì Huế đã được ghi từ trước, nhưng cho thấy ngưỡng quá chặt sẽ làm mất correction thật.

### 4.3 Entity extraction có cấu trúc
- Fact được lưu theo 8 field cố định (`name`, `location`, `profession`, `response_style`, `interests`, `favorite_drink`, `favorite_food`, `pet`), thay vì ghi tự do. Nhờ vậy có thể upsert theo key, câu hỏi recall được định tuyến theo field (`requested_fields`), và tool live `save_user_fact` có thể kiểm tra key hợp lệ.
- **Rủi ro:** thông tin ngoài 8 field (ví dụ "chạy bộ lúc 6 giờ sáng") bị bỏ qua. Thêm field mới cần sửa code.

## 5. Hạn chế và hướng mở rộng
- Chế độ offline dùng regex và whitelist, chỉ đủ cho bộ dữ liệu này. Ở production nên dùng LLM extraction với output có cấu trúc, rồi vẫn giữ lớp confidence/negation làm guardrail.
- `Response quality` offline là heuristic. LLM-as-judge chỉ chạy ở `--live`, nên giữa hai chế độ cột này không so sánh trực tiếp được.
- Chế độ live: nếu yêu cầu `--live` mà agent không khởi tạo được (thiếu key, lỗi SDK…), benchmark dừng và in lý do, chứ không lặng lẽ chạy offline. Token lấy từ `usage_metadata` của **mọi** lần gọi model trong một lượt (gồm cả tool call). Còn hai giới hạn: lượt gọi model của `SummarizationMiddleware` không được tính vào token, và cột `Compactions` ở chế độ live đếm theo `CompactMemoryManager` chạy song song (cùng ngưỡng), chứ không phải số lần middleware thật sự tóm tắt.
- Bộ Guardrail còn nhỏ (4 câu hỏi), nên dùng để chứng minh cơ chế hoạt động, chưa đủ để ước lượng tỉ lệ lỗi.
- Memory decay (giảm ưu tiên fact lâu không nhắc lại) chưa làm. Đây là bước tiếp theo để chặn các field danh sách phình to.

## 6. Cách chạy

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python src/benchmark.py            # offline, deterministic
python src/benchmark.py --verbose  # in cả câu hỏi và câu trả lời recall
python src/benchmark.py --live     # dùng provider trong .env (xem .env.example) + LLM judge
python src/ablation.py             # tái tạo các bảng ablation ở mục 3.3 và 4
pytest src/test_agents.py -v       # 17 test
```
