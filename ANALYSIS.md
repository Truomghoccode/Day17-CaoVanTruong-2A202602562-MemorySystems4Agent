# Báo Cáo Phân Tích Chuyên Sâu: Memory Systems for AI Agent
**Tác giả:** Cao Văn Trường - 2A202602562  
**Học phần:** Phase 2, Track 3, Day 17: Memory Systems for AI Agent  
**Môi trường thực nghiệm:** Python 3.11.16 (`conda env: aivn`), Pytest 8.3.5, LangChain 1.4.2  

---

## 1. Bảng Số Liệu Thực Nghiệm Thu Được Từ Benchmark

Dưới đây là số liệu đo lường trực tiếp từ việc thực thi `python src/benchmark.py` trên cùng bộ dữ liệu chuẩn:

### Bảng 1: Standard Benchmark (`data/conversations.json` - 10 hội thoại, user `dungct`)
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 1,335 | **16,575** | 0.0% | 20.0% | 0 | 0 |
| **Advanced** | 2,540 | 29,898 | **100.0%** | **100.0%** | 301 | 0 |

### Bảng 2: Long-Context Stress Benchmark (`data/advanced_long_context.json` - 16 lượt dài, user `dungct_stress`)
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 221 | 25,314 | 0.0% | 20.0% | 0 | 0 |
| **Advanced** | 1,046 | **10,714** | **100.0%** | **100.0%** | 229 | **7** |

---

## 2. Luồng Logic 5 Mắt Xích Theo Rubric.md

Rubric yêu cầu reviewer phải thấy rõ một chuỗi logic chặt chẽ, trong đó mỗi luận điểm đều được chống đỡ bởi số liệu cụ thể, cơ chế trong mã nguồn `src/` và đánh đổi kỹ thuật đi kèm:

```
[Mắt xích 1: Baseline không nhớ dài hạn]
      ↓ (chứng minh bằng Recall 0.0% và Memory growth 0 bytes)
[Mắt xích 2: Advanced thêm User.md nên recall tăng]
      ↓ (chứng minh bằng Recall 100.0% và Memory growth 301/229 bytes)
[Mắt xích 3: Hội thoại dài làm prompt cost của Baseline bùng nổ]
      ↓ (chứng minh bằng 25,314 Prompt tokens ở Baseline)
[Mắt xích 4: Compact Memory kéo chi phí ngữ cảnh xuống]
      ↓ (chứng minh bằng 10,714 Prompt tokens và 7 Compactions ở Advanced)
[Mắt xích 5: Hệ thống mạnh hơn nhưng phức tạp hơn, cần Guardrail]
      ↓ (chứng minh bằng chênh lệch Agent tokens và cơ chế Conflict/Noise Handling)
```

### Mắt xích 1: Baseline không nhớ dài hạn
- **Số liệu chứng minh:** Cột `Cross-session recall` của Baseline bằng **0.0%** ở cả hai bảng; `Memory growth (bytes)` bằng **0** tuyệt đối.
- **Cơ chế trong code:** Trong [src/agent_baseline.py](file:///c:/Users/caotr/OneDrive/Documents/Day17-CaoVanTruong-2A202602562-MemorySystems4Agent/src/agent_baseline.py), BaselineAgent chỉ khởi tạo một dictionary bộ nhớ trong RAM `self.sessions[thread_id]`. Khi benchmark kích hoạt câu hỏi kiểm tra tại thread mới (`conv-xx-recall`), thread này hoàn toàn rỗng. Do không có tệp lưu trữ trên đĩa, Baseline không thể liên kết danh tính người dùng và buộc phải trả lời: *"Xin lỗi bạn, tôi chưa có thông tin về bạn trong phiên trò chuyện mới này."*
- **Đánh đổi / Giới hạn:** Thiết kế này cực kỳ đơn giản và an toàn về mặt cô lập dữ liệu (không sợ rò rỉ dữ liệu giữa các phiên), nhưng hoàn toàn thất bại trong các ứng dụng thực tế đòi hỏi cá nhân hóa người dùng dài hạn.

### Mắt xích 2: Advanced thêm `User.md` nên recall tăng
- **Số liệu chứng minh:** Cột `Cross-session recall` của Advanced nhảy vọt lên **100.0%**; `Response quality` đạt **100.0%**; đồng thời `Memory growth (bytes)` ghi nhận tăng **301 bytes** (Standard) và **229 bytes** (Stress).
- **Cơ chế trong code:** Trong [src/agent_advanced.py](file:///c:/Users/caotr/OneDrive/Documents/Day17-CaoVanTruong-2A202602562-MemorySystems4Agent/src/agent_advanced.py), mỗi lượt trao đổi đều đi qua `extract_profile_updates()` trong [src/memory_store.py](file:///c:/Users/caotr/OneDrive/Documents/Day17-CaoVanTruong-2A202602562-MemorySystems4Agent/src/memory_store.py) để bóc tách facts, sau đó gọi `UserProfileStore.upsert_facts()` ghi bền vững xuống đĩa tại `state/profiles/<user>/User.md`. Khi bước sang thread recall mới, hàm `_offline_response()` nạp facts từ `User.md` để trả lời chính xác tên, nghề nghiệp mới, nơi ở hiện tại và sở thích.
- **Đánh đổi / Giới hạn:** Tăng chi phí I/O ghi đĩa và tiêu tốn thêm tài nguyên phân tích chuỗi văn bản trên mỗi lượt tương tác.

### Mắt xích 3: Hội thoại dài làm prompt cost của Baseline tăng mạnh
- **Số liệu chứng minh:** Tại bảng Stress Benchmark, cột `Prompt tokens processed` của Baseline tăng vọt lên **25,314 tokens** chỉ sau 16 lượt trao đổi dài, cao gấp hơn 1.5 lần tổng chi phí prompt của cả 10 hội thoại trong Standard Benchmark gộp lại (16,575 tokens).
- **Cơ chế trong code:** Trong `BaselineAgent._reply_offline()`, Baseline dồn toàn bộ lịch sử hội thoại trước đó vào prompt: `full_prompt = f"{history_text}\nuser: {message}"`. Chi phí xử lý tích lũy tăng theo hàm bậc hai:
  $$\text{Total Prompt Tokens} = \sum_{i=1}^{N} \text{Tokens}(\text{Turn}_i) \approx O(N^2)$$
- **Đánh đổi / Giới hạn:** Nguy cơ vượt trần Context Window của LLM, tăng mạnh độ trễ Time-to-First-Token (TTFT) và làm bùng nổ chi phí API hóa đơn hàng tháng.

### Mắt xích 4: Compact Memory kéo chi phí ngữ cảnh xuống
- **Số liệu chứng minh:** Tại bảng Stress, `Prompt tokens processed` của Advanced chỉ còn **10,714 tokens** (tiết kiệm **14,600 tokens**, tương đương **giảm 57.7%** so với Baseline); số lần nén `Compactions` đạt **7 lần**.
- **Cơ chế trong code:** [src/memory_store.py](file:///c:/Users/caotr/OneDrive/Documents/Day17-CaoVanTruong-2A202602562-MemorySystems4Agent/src/memory_store.py) triển khai `CompactMemoryManager.append()`. Khi tổng token của thread vượt qua `compact_threshold_tokens = 800`, hệ thống trích xuất các tin nhắn cũ qua `summarize_messages()`, tích lũy vào `summary` không trùng lặp và chỉ giữ lại `compact_keep_messages = 4` tin nhắn gần nhất. Hàm `_estimate_prompt_context_tokens()` chỉ phải nạp:
  $$\text{Payload} = \text{Tokens}(\text{User.md}) + \text{Tokens}(\text{Summary}) + \text{Tokens}(\text{4 recent messages}) \approx O(N \cdot K)$$
- **Đánh đổi / Giới hạn:** Nén thông tin có thể làm mờ nhạt hoặc mất các chi tiết số liệu nhỏ (loss of nuances) nếu chúng không thuộc diện được bộ tóm tắt ghi nhận.

### Mắt xích 5: Hệ thống mạnh hơn nhưng phức tạp hơn, cần Guardrail tốt hơn
- **Số liệu chứng minh:** `Agent tokens only` của Advanced tăng lên **2,540 tokens** (Standard) và **1,046 tokens** (Stress) so với Baseline (1,335 và 221); kiến trúc phình to từ 1 class naive lên 3 class chuyên trách (`UserProfileStore`, `CompactMemoryManager`, `AdvancedAgent`).
- **Cơ chế trong code:** Phản hồi của Advanced giàu thông tin và tuân thủ format bullet hơn hẳn, dẫn tới token đầu ra cao hơn. Tuy nhiên, nếu không có bộ lọc Guardrail (`is_pure_query` và lọc nhiễu trong `extract_profile_updates`), các tin nhắn chứa câu đùa hoặc câu hỏi kiểm tra sẽ bị lưu nhầm thành fact, làm hỏng toàn bộ hồ sơ `User.md`.
- **Đánh đổi / Giới hạn:** Đòi hỏi các bài kiểm thử nghiêm ngặt ([src/test_agents.py](file:///c:/Users/caotr/OneDrive/Documents/Day17-CaoVanTruong-2A202602562-MemorySystems4Agent/src/test_agents.py)) để đảm bảo logic compact và đồng bộ tệp không gây race-condition hoặc làm hỏng dữ liệu.

---

## 3. Trả Lời 4 Câu Hỏi Trọng Tâm Của Bước 8 (Guide.md)

### 3.1. Vì sao Advanced có recall tốt hơn Baseline?
1. **Số liệu:** Cột `Cross-session recall` của Advanced đạt **100.0%** ở cả hai bảng Standard và Stress, trong khi Baseline đạt **0.0%**.
2. **Cơ chế trong code:** Fact của người dùng đi qua luồng khép kín:
   $$\text{User Input} \xrightarrow{\text{extract\_profile\_updates()}} \text{Fact Dict} \xrightarrow{\text{UserProfileStore.upsert\_facts()}} \text{User.md (Disk)} \xrightarrow{\text{\_offline\_response()}} \text{Answer}$$
   Baseline chỉ lưu trong biến session nội bộ của `thread_id` cũ, nên khi benchmark mở thread `conv-xx-recall` độc lập, Baseline có 0 token ngữ cảnh quá khứ.
3. **Giới hạn đi kèm:** Độ chính xác của recall phụ thuộc hoàn toàn vào chất lượng trích xuất fact. Nếu bộ trích xuất bắt nhầm thông tin, lỗi sai đó sẽ trở thành "ảo giác vĩnh viễn" (Permanent Hallucination) trong `User.md`.

### 3.2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?
1. **Số liệu:** Ở bảng Standard Benchmark, `Prompt tokens processed` của Advanced là **29,898 tokens**, cao hơn **80.4%** so với Baseline (**16,575 tokens**); `Agent tokens only` cũng cao hơn (**2,540 vs 1,335 tokens**).
2. **Cơ chế trong code:** Trong 10 hội thoại thông thường, mỗi hội thoại chỉ có 10 lượt ngắn, chưa bao giờ vượt ngưỡng 800 tokens để kích hoạt nén (`Compactions = 0`). Tuy nhiên, ở **mỗi lượt**, Advanced Agent đều phải nạp toàn bộ nội dung `User.md` thông qua `_estimate_prompt_context_tokens()`. Đây chính là "khoản thuế cố định" (Memory Overhead Tax) để duy trì khả năng nhận thức danh tính.
3. **Giới hạn đi kèm:** Với các tác vụ chat 1 lần (ephemeral/transactional tasks) nơi người dùng không quay lại, việc nạp persistent memory tạo ra lãng phí token thuần túy mà không đem lại giá trị hồi đáp tương xứng.

### 3.3. Vì sao Compact có lợi thế ở hội thoại dài?
1. **Số liệu:** Ở bảng Long-Context Stress Benchmark, `Prompt tokens processed` của Advanced giảm xuống **10,714 tokens**, tiết kiệm tới **57.7%** (giảm 14,600 tokens) so với mức **25,314 tokens** của Baseline.
2. **Cơ chế trong code:** Cần phân định rạch ròi: **Compact Memory tối ưu hóa cột `Prompt tokens processed`, hoàn toàn KHÔNG tối ưu hóa `Agent tokens only`**.
   - Cột `Agent tokens only` thực tế tăng từ 221 lên 1,046 vì agent phải trả lời chi tiết và tuân thủ định dạng 3 bullet.
   - Cột `Prompt tokens processed` giảm sâu vì `CompactMemoryManager` đã kích hoạt **7 lần compactions**, đều đặn nén hàng nghìn ký tự tin tức dài về Artemis III, X-59, WMO El Nino thành các gạch đầu dòng tóm tắt súc tích, giải phóng bộ nhớ đệm và chỉ giữ lại 4 tin nhắn gần nhất.
3. **Giới hạn đi kèm:** Chi phí tính toán để tạo bản tóm tắt tích lũy. Nếu ngưỡng token đặt quá thấp, hiện tượng nén quá sớm (Over-compaction) sẽ làm đứt gãy tính liên tục của cuộc trò chuyện.

### 3.4. File memory tăng trưởng ra sao và rủi ro gì đi kèm?
1. **Số liệu:** Cột `Memory growth (bytes)` tăng **301 bytes** trong Standard Benchmark và **229 bytes** trong Stress Benchmark; tương ứng với **7 lần Compactions** trong stress test.
2. **Cơ chế trong code:** Tệp `User.md` được quản lý theo mô hình Key-Value Markdown (`- key: value`). Do cơ chế `upsert_facts` cập nhật đè lên các khóa trạng thái cũ, kích thước tệp được chặn trần hiệu quả, không bị tăng vô hạn theo số lượt hội thoại.
3. **Rủi ro cụ thể quan sát được:**
   - *Rủi ro Memory Poisoning (Nhiễm độc bộ nhớ):* Nếu người dùng nói đùa hoặc đưa ra giả định sai mà hệ thống không có bộ lọc, fact sai sẽ bị ghi vào `User.md` và đầu độc tất cả các phiên tương lai.
   - *Rủi ro Nuance Flattening (Mất sắc thái):* Sau 7 lần compact, các sắc thái cảm xúc hoặc chi tiết phụ không nằm trong danh mục tóm tắt chính sẽ bị triệt tiêu hoàn toàn.

---

## 4. Phân Tích Tính Năng Mở Rộng Đạt Mốc 90–100 Điểm (Bonus Deep Dive)

Trong 4 hướng mở rộng được quy định trong Rubric.md, dự án đã chọn và triển khai hoàn chỉnh hướng: **Conflict Handling & Noise/Question Guardrail khi có Correction mới**.

### 4.1. Bonus đó giải quyết vấn đề gì?
1. **Xung đột khi người dùng đính chính (Fact Correction):**
   - Nơi ở: Ban đầu ở `Đà Nẵng` $\rightarrow$ đính chính chuyển về `Huế` $\rightarrow$ trong stress test lại chuyển làm việc tại `Đà Nẵng`.
   - Nghề nghiệp: Ban đầu làm `backend engineer` $\rightarrow$ đính chính đổi sang `MLOps engineer`.
   - *Vấn đề:* Nếu lưu trữ ngây thơ, agent sẽ giữ cả 2 thông tin trái ngược hoặc ưu tiên sai dữ liệu cũ.
2. **Nguy cơ lưu nhầm câu hỏi thành Fact (Query Misinterpretation):**
   - Khi người dùng hỏi: *"Hiện tại mình đang ở đâu?"* hoặc *"Tên mình là gì?"*, các regex thông thường rất dễ bóc tách chính từ trong câu hỏi thành fact rác.
3. **Nhiễu thông tin và câu nói đùa (Noise & Sarcasm):**
   - Người dùng đùa: *"hay là chuyển sang product manager cho đỡ phải ngồi canh pipeline, nhưng đó chỉ là câu đùa."*
   - Người dùng nhắc địa danh công tác: *"Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày với đối tác chứ không phải nơi ở hiện tại."*

### 4.2. Cơ chế cải thiện Recall và Token Cost như thế nào?
- **Về mặt kỹ thuật:**
  - Trong `extract_profile_updates()`: Cài đặt cờ `is_pure_query` phát hiện các câu hỏi truy vấn để bỏ qua việc trích xuất.
  - Cài đặt bộ lọc ngữ cảnh: Bỏ qua `product manager` khi xuất hiện từ khóa `"đùa"`, `"câu đùa"`; bỏ qua `Hà Nội` khi đi kèm `"họp"`, `"chứ không phải nơi ở"`.
  - Phân loại dữ liệu: Với các fact trạng thái duy nhất (Location, Profession), `UserProfileStore.upsert_facts()` thực hiện ghi đè giá trị mới nhất. Với các preference tích lũy (Style, Interests), hệ thống thực hiện hợp nhất tập hợp (Set Union).
- **Hiệu quả định lượng:**
  - Đưa `Cross-session recall` trên các câu hỏi hóc búa (như câu hỏi đính chính ở conv-03, conv-06, conv-10 và stress test) đạt điểm tuyệt đối **100.0%**.
  - Giữ dung lượng `User.md` gọn gàng ở mức **301 / 229 bytes**, ngăn chặn rác dữ liệu làm phình to `Prompt tokens processed`.

### 4.3. Bonus đó tạo thêm rủi ro gì cho hệ thống?
Không có giải pháp nào là miễn phí, cơ chế này đem lại 3 rủi ro kỹ thuật:
1. **Rủi ro False Negatives (Bỏ sót thông tin thật):** Nếu người dùng thực sự chuyển sang làm "Product Manager" nhưng lại dùng từ "đùa" trong cùng một câu theo ngữ cảnh khác, bộ lọc từ khóa có thể bỏ qua một fact hợp lệ.
2. **Thiếu cơ chế Temporal Rollback (Không có lịch sử phiên bản):** Việc ghi đè hoàn toàn giá trị cũ đồng nghĩa với việc hệ thống không lưu lại timeline lịch sử thay đổi (ví dụ: người dùng từng ở Đà Nẵng trước khi về Huế). Nếu một đính chính bị trích xuất nhầm, fact cũ đúng sẽ bị xóa vĩnh viễn mà không thể rollback.
3. **Độ phức tạp bảo trì Rule-based:** Khi mở rộng sang đa ngôn ngữ hoặc ngữ cảnh tự nhiên phức tạp, việc duy trì các mẫu regex và quy tắc loại trừ trở nên giòn gãy (fragile), đòi hỏi phải chuyển dịch sang LLM-based Extraction với JSON schema và Confidence Score trong môi trường production lớn.

---

## 5. Bảng Tự Đánh Giá Đối Chiếu Mốc Điểm Rubric.md

| Mốc điểm | Tiêu chí Rubric | Trạng thái dự án | Minh chứng thực tế |
| :---: | :--- | :---: | :--- |
| **0 – 60** | Đủ Baseline, Advanced có User.md, có compact, benchmark tiếng Việt, cấu trúc repo chuẩn. | **ĐẠT** | Đầy đủ mã nguồn trong `src/`, dữ liệu trong `data/`, cấu trúc module hóa phân tầng rõ ràng. |
| **60 – 75** | Benchmark chạy cùng input cho cả 2 agent; có test cho User.md, compact trigger, cross-session recall; bảng đủ 6 cột. | **ĐẠT** | `pytest src/test_agents.py -v` xanh 4/4 bài test; bảng benchmark in chuẩn xác 6 cột bắt buộc. |
| **75 – 90** | Có cả Standard và Stress; stress lộ rõ chi phí ngữ cảnh baseline; phân tích rõ compact tối ưu `Prompt tokens processed`. | **ĐẠT** | Stress test chứng minh Baseline tốn 25,314 tokens; Advanced tiết kiệm 57.7% prompt tokens; phân tích tường minh hai loại tokens. |
| **90 – 100** | Có bonus hữu ích (Conflict Handling / Noise Filtering) kèm đủ 3 câu trả lời: giải quyết gì, cải thiện gì, rủi ro gì. | **ĐẠT** | Trình bày toàn diện cơ chế giải quyết đính chính, lọc nhiễu, đối chiếu số liệu và mổ xẻ 3 rủi ro kỹ thuật thực tế. |
