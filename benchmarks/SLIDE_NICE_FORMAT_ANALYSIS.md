# Phân tích bảng nice_format — opt-cpu-async-merged1.0 vs baselines

> Cấu hình "Fcfs + Cpu rank (ours)": T_warmup=1s, chunk_size=4 mặc định.

---

## Slide TTFT — Phân tích

Trên cả ba thống kê mean, median và p99, ours đều thua Ranking scheduler (paper) ở mọi rate. Điểm đáng chú ý là khoảng cách lớn nhất không nằm ở rate cao như nhiều người dự đoán, mà lại nằm ở rate thấp và rate trung bình — nơi mọi scheduler khác đều chạy thoải mái thì ours lại là người duy nhất gặp khó.

Ở rate=2, ours đạt mean TTFT 643 ms, gấp gần ba lần paper (229 ms) và gấp 3.6 lần SRTF (177 ms). Đáng nói hơn, FCFS đơn giản chỉ đạt 209 ms — tức là ours kém ngay cả so với scheduler cơ bản nhất. Pattern lặp lại y hệt ở rate=4: ours 1350 ms, paper 365 ms, FCFS 409 ms, SRTF 290 ms. Khi queue ngắn và mọi phương pháp khác chạy nhẹ nhàng, ours lại trả phí cao nhất. Lý do là với T_warmup chỉ 1 giây, scheduler chuyển sang chế độ scoring rất sớm, mỗi request mới đến phải có score thực mới được vào hàng chạy, nên thời gian chờ predictor cộng thẳng vào TTFT của từng request. Ở rate thấp queue ngắn, predictor không có cơ hội amortize chi phí này — chi phí thừa xuất hiện trên đầu mỗi request, và đó là điểm yếu mà ours không thể giấu được.

Vùng rate=8 là nơi tỉ đối giữa ours và paper xấu nhất. Mean TTFT của ours đạt 25596 ms còn paper chỉ 8311 ms — chênh hơn ba lần. Median còn rộng hơn nữa: ours 19753 ms so với paper 546 ms — gấp 36 lần. Đây là rate mà queue đã đủ dài để gây tắc nghẽn nhưng GPU model chính chưa bão hòa, đúng vùng predictor trở thành bottleneck thực sự. Hệ thống không kịp score những request mới đến nhanh hơn khả năng xử lý của predictor, backlog tích tụ, các request bị giam lại không vào được hàng chạy, và TTFT của chúng bùng nổ.

Khi rate vượt 16, khoảng cách thu hẹp đáng kể. Ở r=16 ours kém paper chỉ 6%, ở r=32 kém 11%, đến r=64 lại nới ra thành 26%. Đây là tín hiệu rằng ở rate cao GPU model chính trở thành bottleneck dominant, mọi scheduler đều bị ép phải chờ KV cache, nên khoảng cách giữa các phương pháp bị nén lại. Ours không hơn được paper ở vùng này nhưng cũng không thua quá xa — bottleneck của hệ thống đã chuyển sang chỗ khác và che đi hạn chế của predictor.

So với SRTF — vốn là oracle có ground-truth output length — ours thua ở mọi rate, thường từ 1.4 đến 2.5 lần. SRTF là baseline trần, và khoảng cách giữa ours với SRTF cho thấy còn rất nhiều dư địa cải thiện chất lượng predictor: nếu predictor chính xác bằng oracle, ours hoàn toàn có thể tiệm cận SRTF.

Tổng quan, TTFT của ours thua paper toàn diện. Ở rate thấp và trung bình vấn đề là chi phí scoring không được amortize và predictor backpressure ở r=8. Ở rate cao, ours về gần paper vì bottleneck chuyển sang GPU model chính, nhưng vẫn không thắng được. Khoảng cách với SRTF cho thấy biên cải thiện còn lớn nếu nâng được chất lượng và throughput predictor.

---

## Slide TPOT — Phân tích

TPOT của ours vẽ ra một bức tranh hoàn toàn khác TTFT. Không còn câu chuyện "thua toàn diện" như ở TTFT — ở đây ours đồng thời vừa thắng vừa thua, tùy bạn nhìn vào thống kê nào.

Điểm gây bất ngờ đầu tiên: từ rate=16 trở lên, median TPOT của ours là tốt nhất trong tất cả 5 scheduler, thắng cả SRTF oracle. Ở rate=32, median ours 1308 ms — thấp hơn FCFS (1393), SJF (1394), SRTF (1379), và paper (1365). Ở rate=64 cũng vậy: 1354 ms, thấp hơn mọi baseline. Điều này nghĩa là với 50% request "phía dưới" — tức là những request ngắn — ours đang chạy nhanh hơn cả oracle. Nói cách khác, đa số request được hưởng lợi rõ rệt từ chính sách ranking của ours.

Nhưng p99 lại kể một câu chuyện hoàn toàn ngược lại. Ở rate=32, p99 TPOT của ours đạt 11853 ms — gấp 2.4 lần paper (4848) và gấp 2.9 lần SRTF (4100). Ở rate=64 còn nặng hơn: 12722 ms so với paper 4891 và SRTF 3855, tức gấp 3.3 lần SRTF. Trong khi mean và median ours còn ngang ngửa hoặc thắng paper, thì p99 nói rằng có một nhóm request — khoảng 1% cuối — phải trả giá rất đắt.

Pattern "median thắng, p99 catastrophic" này chính là dấu hiệu kinh điển của starvation tail — hiện tượng đói tài nguyên ở đuôi phân phối. Đây là biểu hiện rất quen thuộc với một scheduler dùng SJF mà không có cơ chế bảo vệ long job. Khi queue bão hòa ở rate cao, short job liên tục được đẩy lên đầu hàng vì có score cao hơn (predictor đoán "ngắn"). Mỗi lần KV cache trống ra một slot, scheduler chọn short job mới đến chứ không phải long job đã chờ từ lâu. Long job ấy đứng yên, đôi khi đã chạy được vài token nhưng bị swap ra ngoài khi short job mới chen vào. Kết quả là short jobs kéo median xuống thấp đẹp đẽ, trong khi long jobs bị bỏ đói kéo p99 lên cao chót vót.

So sánh với FCFS làm bật lên trade-off này rõ nhất. FCFS có p99 TPOT chỉ 1855 ms ở rate=64 — nhỏ hơn ours gần bảy lần — vì FCFS không bao giờ "chen ngang" long job: ai đến trước chạy trước, không ai bị bỏ đói. Cái giá FCFS phải trả là mean và median TPOT cao hơn ours một chút (1411 ms so với 1354 ms). Đây là lựa chọn chính sách: FCFS đảm bảo công bằng tuyệt đối nhưng chậm cho mọi người; ours nhanh cho đa số nhưng tàn nhẫn với thiểu số.

So sánh với SRTF còn cho thấy thêm một lớp khác. SRTF cũng SJF-style nhưng với ground truth nên cũng có tail tăng so với FCFS — p99 SRTF 3855 ms ở rate=64 (gấp đôi FCFS). Đây là cái giá tự nhiên của SJF: bất kỳ scheduler nào ưu tiên short job đều có tail kéo dài. Nhưng ours đẩy tail xa hơn SRTF gấp 3.3 lần. Phần "vượt mức" này đến từ predictor noise: predictor đôi khi gán score cao cho job thực sự dài (tức là đoán nhầm "ngắn" cho một job dài), khiến job đó bị admit "nhầm" rồi bị kéo dài thời gian chạy giữa hàng đống short job khác — một sai lầm mà oracle SRTF không bao giờ mắc phải.

So với paper, ours rõ ràng đang ở thế bất lợi trên p99. Paper cũng dùng cách ranking nhưng giữ p99 ở mức chỉ gấp 1.3 lần SRTF (4891 vs 3855 ở rate=64), trong khi ours gấp 3.3 lần. Khoảng cách này gợi ý paper đã có cơ chế gì đó để cân bằng giữa median và tail — có thể là starvation control, có thể là batch policy khác — và đây chính là điểm ours còn cần học hỏi.

Tổng quan, TPOT của ours thắng median bằng cách hy sinh tail. Short jobs hưởng lợi rõ rệt và đây là điểm mạnh đáng giữ. Nhưng p99 catastrophic ở rate cao là vấn đề nghiêm trọng, không thể đem ra production với SLO p99. So sánh với SRTF cho thấy ngay cả oracle cũng có tail tăng, nhưng predictor noise của ours làm tail tệ thêm gấp ba lần. So sánh với paper cho thấy hoàn toàn có thể đạt được trade-off tốt hơn giữa median và p99 nếu có cơ chế bảo vệ long job phù hợp.

---

## Đối chiếu hai slide

Hai metric vẽ ra hai câu chuyện rất khác nhau về ours. TTFT thua paper toàn diện ở mọi thống kê và mọi rate, không có vùng nào thắng — vấn đề chủ yếu là chi phí scoring không được amortize ở rate thấp và predictor backpressure ở rate trung. TPOT lại có pattern win-median-lose-p99: short jobs hưởng lợi (median thắng cả oracle), nhưng long jobs trả giá thảm họa ở tail. Hai vấn đề này có hai nguyên nhân khác nhau và cần hai hướng cải thiện khác nhau: TTFT cần giảm chi phí scoring overhead, còn TPOT cần cơ chế bảo vệ long job để khắc phục starvation.
