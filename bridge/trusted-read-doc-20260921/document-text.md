# 專題彙報

- Document ID: 1NxMLiXVmdF1IsxHXSJ_VLtgyyES7edLc3HX0W-Fn3VY
- Revision ID: ANLCKQmEB3H2lWJ8870dw5F_CE-KAZpXVYPPKZwoFQSC-fe6qCFPLzAIFDSbBf99XMX893nnVB6Je-nGmEEORXrRMtI2qZOsizWtLlpID0Y
- Selected tab: all
- Protected controls: 0
- Opaque controls: 0
- Authoritative dropdowns: 0

Protected-control annotations are preservation instructions. Do not insert their displayed placeholder text to recreate a native control.

## Overview (t.0)

[P00001 | 1:2 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00002 | 2:3 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00003 | 3:4 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00004 | 4:5 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00005 | 5:14 | NORMAL_TEXT]
環境維護科技執法

[P00006 | 14:21 | NORMAL_TEXT]
專題結果報告

[P00007 | 21:22 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00008 | 22:23 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00009 | 23:24 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00010 | 24:25 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00011 | 25:34 | NORMAL_TEXT]
指導老師：熊博安

[P00012 | 34:35 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00013 | 35:36 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00014 | 36:50 | NORMAL_TEXT]
412410068 張宇誠

[P00015 | 50:64 | NORMAL_TEXT]
412410083 張哲誠

[P00016 | 64:65 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00017 | 66:74 | NORMAL_TEXT]
（一）摘要	1

[P00018 | 74:87 | NORMAL_TEXT]
（二）研究動機與問題	1

[P00019 | 87:102 | NORMAL_TEXT]
（三）研究方法與技術說明	1

[P00020 | 102:109 | NORMAL_TEXT]
系統架構	1

[P00021 | 109:120 | NORMAL_TEXT]
資料集整理與訓練	2

[P00022 | 120:130 | NORMAL_TEXT]
RT-DETR	3

[P00023 | 130:150 | NORMAL_TEXT]
YOLO-Segmentation	4

[P00024 | 150:172 | NORMAL_TEXT]
YOLO-Pose & STGCN++	5

[P00025 | 172:190 | NORMAL_TEXT]
反追蹤模組 Backtrack	6

[P00026 | 190:205 | NORMAL_TEXT]
Optimization	7

[P00027 | 205:216 | NORMAL_TEXT]
（四）成果與總結	8

[P00028 | 216:226 | NORMAL_TEXT]
（五）引用文獻	8

[P00029 | 227:233 | HEADING_1]
（一）摘要

[P00030 | 233:234 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00031 | 234:546 | NORMAL_TEXT]
本研究針對低幀率監視影像中的微小垃圾辨識與環境違規行為，建立一套智慧環境違規事件偵測系統。由於煙蒂、衛生紙與塑膠袋等垃圾在畫面中尺寸極小，且低幀率會造成物體在相鄰影格間位移較大，因此一般物件偵測方法容易產生漏判。本研究以 RT-DETR 為基礎，加入 P2 高解析度特徵層，並融合 RGB 影像與前後影格的 Pixel Change Map，以提升模型對微小移動物體的辨識能力。實驗結果顯示，加入影格變化資訊後，模型的 percision 度與 recall 均有所提升，並能結合行為辨識技術，擴展系統對不同環境違規事件的偵測能力。此外，本研究利用 ST-GCN 分析人體骨架關節在時間與空間上的變化，以判斷隨地便溺行為。

[P00032 | 546:547 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00033 | 547:558 | HEADING_1]
（二）研究動機與問題

[P00034 | 558:559 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00035 | 559:728 | NORMAL_TEXT]
近年來，非法傾倒廢棄物、隨手丟棄垃圾的違法現象頻傳，有些人甚至會專門騎 車到定點丟棄家庭垃圾。許多違規者抱持僥倖心理，認為從車內丟棄垃圾的過程僅有 一瞬間，難以被察覺或取締。然而此類行為不僅破壞市容與環境衛生，更可能造成溝 渠堵塞。不過由於環保機關受限於人力與行政成本，難以針對煙蒂或小型垃圾逐一調 閱監視器進行開罰，導致執法效率受限。

[P00036 | 728:799 | NORMAL_TEXT]
 有鑑於此，本研究旨在導入結合人工智慧的科技執法系統，以克服傳統人力稽查成本過高的問題，有效填補過去難以偵測的小型垃圾違規行為精準落實罰則。

[P00037 | 799:800 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00038 | 800:813 | HEADING_1]
（三）研究方法與技術說明

[P00039 | 813:814 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00040 | 814:820 | HEADING_2]
系統架構 

[P00041 | 820:839 | NORMAL_TEXT]
圖一、系統pipeline簡易架構圖

[P00042 | 839:840 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00043 | 840:1031 | NORMAL_TEXT]
本系統先以 YOLO-Seg 偵測並追蹤車輛與機車，經車輛閘門篩選後，分成垃圾與動作兩條分析分支。垃圾分支利用 RT-DETR 偵測垃圾，透過跨影格追蹤確認事件；動作分支由 YOLO-Pose 擷取人體骨架，再由 ST-GCN 分析骨架序列，辨識違規動作。確認事件後，分別進行垃圾來源反追蹤或違規人物的人車關聯回溯，整合結果並執行車牌偵測與 OCR，最後輸出標註影片及事件報告。 

[P00044 | 1031:1212 | NORMAL_TEXT]
加速方面，pipeline 採背景讀片、主執行緒推論與背景寫片，透過 Queue 讓不同影格的讀取、分析與編碼重疊執行，減少等待並限制記憶體用量。讀片階段預先建立運動遮罩及四通道模型輸入，主流程搭配批次推論、YOLO-Seg 跳幀快取與車輛閘門，降低重複運算。此外，垃圾反追蹤與車牌辨識可派至背景處理；逐幀狀態更新仍維持時間順序，以確保軌跡與事件資料的一致性。

[P00045 | 1212:1213 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00046 | 1213:1222 | HEADING_2]
資料集整理與訓練

[P00047 | 1222:1317 | NORMAL_TEXT]
	我們與嘉義縣環保局合作取得資料集，資料集內為路上 CCTV 影像片段，包含亂丟煙蒂、飲料杯、大型垃圾袋、隨地大小便等違規影像。大部分錄像為固定鏡頭拍攝紅綠燈路口、路肩等違規垃圾丟棄熱點。

[P00048 | 1317:1340 | NORMAL_TEXT | LIST id=kix.lvr5cvagrao2 level=0]
影片橫跨三個月份2024年的6月、7月、9月

[P00049 | 1340:1366 | NORMAL_TEXT | LIST id=kix.lvr5cvagrao2 level=0]
影片可分為兩類 199部亂丟垃圾、141部隨地便溺

[P00050 | 1366:1399 | NORMAL_TEXT | LIST id=kix.lvr5cvagrao2 level=0]
監視器型號不同分為10fps、12fps及30fps三種幀數影片

[P00051 | 1399:1401 | NORMAL_TEXT]
[INLINE_OBJECT kix.uodr4h5sr91x]

[P00052 | 1401:1421 | NORMAL_TEXT]
圖一、亂丟垃圾（左）及隨地大小便（右）

[P00053 | 1421:1422 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00054 | 1422:1430 | NORMAL_TEXT]
A.垃圾資料集

[P00055 | 1430:1431 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00056 | 1431:1574 | NORMAL_TEXT]
	我們參考此論文[]的方式，先將每部影片中有垃圾掉落的片段擷取下5～10幀，沒有垃圾掉落的片段，如：車輛經過（包含汽車、貨車、腳踏車）、行人經過的背景片段我們也會擷取2~5幀。不過因為部份影片的垃圾掉落速度急快，可能在十幀內垃圾就已消失在草叢中，此時我們也會調整背景資料所擷取的幀數。

[P00057 | 1574:1593 | NORMAL_TEXT | LIST id=kix.v5kqmivrwgy4 level=0]
199部影片總共擷取成1632張照片

[P00058 | 1593:1637 | NORMAL_TEXT | LIST id=kix.v5kqmivrwgy4 level=0]
採用 YOLO 資料集格式標註，以Bounding Box標示垃圾在影像中的位置及範圍

[P00059 | 1637:1660 | NORMAL_TEXT | LIST id=kix.v5kqmivrwgy4 level=0]
有垃圾資料:背景資料  比例為約 2:1  

[P00060 | 1660:1684 | NORMAL_TEXT | LIST id=kix.v5kqmivrwgy4 level=0]
訓練集：驗證集：測試集  比例為約 8:1:1

[P00061 | 1684:1744 | NORMAL_TEXT | LIST id=kix.v5kqmivrwgy4 level=0]
資料集以影片為單位進行劃分；若某部影片被分配至訓練集，則由該影片擷取的所有影格皆會納入訓練集，不會分散至驗證集或測試集

[P00062 | 1744:1745 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00063 | 1745:1747 | NORMAL_TEXT]
[INLINE_OBJECT kix.2vfihkocgctg]

[P00064 | 1747:1768 | NORMAL_TEXT]
圖二、4 Channel 資料產生示意圖

[P00065 | 1768:1769 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00066 | 1769:1987 | NORMAL_TEXT]
	此外，本研究亦使用四通道影像資料，並以 TIFF 點陣圖格式儲存。四通道影像由原始影像的 RGB 三通道與 Pixel Change Map 單通道合併而成。其中，Pixel Change Map 是透過將前一影格與當前影格進行 RGB 像素差分，再轉換為單通道並正規化後所得，如圖○所示。當同一位置的像素顏色在前後影格間的變化越大時，其在 Pixel Change Map 中的像素值也會越高，其數值與RGB一樣在0~255之間。

[P00067 | 1987:1988 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00068 | 1988:1996 | HEADING_2]
RT-DETR

[P00069 | 1999:2005 | NORMAL_TEXT | TABLE row=0 col=0]
Model

[P00070 | 2006:2016 | NORMAL_TEXT | TABLE row=0 col=1]
Percision

[P00071 | 2017:2025 | NORMAL_TEXT | TABLE row=0 col=2]
Recall 

[P00072 | 2026:2032 | NORMAL_TEXT | TABLE row=0 col=3]
mAP50

[P00073 | 2033:2042 | NORMAL_TEXT | TABLE row=0 col=4]
mAP50-95

[P00074 | 2044:2053 | NORMAL_TEXT | TABLE row=1 col=0]
yolo26-l

[P00075 | 2054:2059 | NORMAL_TEXT | TABLE row=1 col=1]
0.59

[P00076 | 2060:2065 | NORMAL_TEXT | TABLE row=1 col=2]
0.36

[P00077 | 2066:2071 | NORMAL_TEXT | TABLE row=1 col=3]
0.39

[P00078 | 2072:2077 | NORMAL_TEXT | TABLE row=1 col=4]
0.22

[P00079 | 2079:2087 | NORMAL_TEXT | TABLE row=2 col=0]
RT-DETR

[P00080 | 2088:2093 | NORMAL_TEXT | TABLE row=2 col=1]
0.73

[P00081 | 2094:2099 | NORMAL_TEXT | TABLE row=2 col=2]
0.27

[P00082 | 2100:2105 | NORMAL_TEXT | TABLE row=2 col=3]
0.32

[P00083 | 2106:2111 | NORMAL_TEXT | TABLE row=2 col=4]
0.17

[P00084 | 2112:2140 | NORMAL_TEXT]
表一、yolo26-l 與 RT-DETR訓練結果比較

[P00085 | 2140:2477 | NORMAL_TEXT]
本研究選用 RT-DETR 作為物件偵測模型。相較於著重推論速度的 YOLO，RT-DETR 採用 Transformer 架構，能整合影像中不同區域的資訊，並以端到端方式直接產生偵測結果，不需要額外進行非極大值抑制（NMS）。其中，基於注意力的同尺度特徵交互（Attention-based Intra-scale Feature Interaction，AIFI）僅在深層的高階特徵上應用自注意力機制，以較低的運算成本捕捉影像中的全域上下文資訊。由於本研究需要辨識菸蒂、檳榔渣等容易被環境雜訊淹沒的微小物件，因此選擇 RT-DETR 作為基礎模型，並進一步加入 P2 高解析度特徵層與 Pixel Change Map，以同時保留微小物件的細節並利用影格間的變化資訊。

[P00086 | 2480:2486 | NORMAL_TEXT | TABLE row=0 col=0]
Model

[P00087 | 2487:2497 | NORMAL_TEXT | TABLE row=0 col=1]
Percision

[P00088 | 2498:2506 | NORMAL_TEXT | TABLE row=0 col=2]
Recall 

[P00089 | 2507:2513 | NORMAL_TEXT | TABLE row=0 col=3]
mAP50

[P00090 | 2514:2523 | NORMAL_TEXT | TABLE row=0 col=4]
mAP50-95

[P00091 | 2525:2529 | NORMAL_TEXT | TABLE row=1 col=0]
RGB

[P00092 | 2530:2535 | NORMAL_TEXT | TABLE row=1 col=1]
0.73

[P00093 | 2536:2541 | NORMAL_TEXT | TABLE row=1 col=2]
0.27

[P00094 | 2542:2547 | NORMAL_TEXT | TABLE row=1 col=3]
0.32

[P00095 | 2548:2553 | NORMAL_TEXT | TABLE row=1 col=4]
0.17

[P00096 | 2555:2578 | NORMAL_TEXT | TABLE row=2 col=0]
RGB + Pixel Change Map

[P00097 | 2579:2584 | NORMAL_TEXT | TABLE row=2 col=1]
0.78

[P00098 | 2585:2590 | NORMAL_TEXT | TABLE row=2 col=2]
0.63

[P00099 | 2591:2596 | NORMAL_TEXT | TABLE row=2 col=3]
0.63

[P00100 | 2597:2602 | NORMAL_TEXT | TABLE row=2 col=4]
0.32

[P00101 | 2603:2649 | NORMAL_TEXT]
表二、RT-DETR RGB(3 channel)與RGB+差幀(4 channel)比較

[P00102 | 2649:2650 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00103 | 2650:2652 | NORMAL_TEXT]
[INLINE_OBJECT kix.9zwioudwzblx]

[P00104 | 2652:2702 | NORMAL_TEXT]
圖三、RT-DETR 4 channel (RGB + Pixel Change Map) 示意圖

[P00105 | 2702:2703 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00106 | 2703:2906 | NORMAL_TEXT]
我們參考文獻Exploiting Temporal Context for Tiny Object Detection[1]，利用影片中的時間資訊改善微小移動物件辨識；同時參考 Leveraging Motion Saliency via Frame Differencing for Enhanced Object Detection in Videos[2]，透過相鄰影格相減取得物體的移動變化資訊。

[P00107 | 2906:2907 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00108 | 2907:3023 | NORMAL_TEXT]
	我們將相鄰影格產生的單通道差幀圖（Pixel Change Map）與原始 RGB 影像串接，形成包含 RGB 三通道及差幀資訊一通道的四通道輸入，並修改 RT-DETR 的輸入層，使模型能同時學習物體的外觀特徵與時間變化特徵。

[P00109 | 3023:3024 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00110 | 3024:3234 | NORMAL_TEXT]
相較於僅使用 RGB 影像，表○顯示加入差幀資訊後，Precision 由 0.73 提升至 0.78，Recall 由 0.27 大幅提升至 0.63；mAP50 亦由 0.32 提升至 0.63，mAP50–90 則由 0.17 提升至 0.32。結果顯示，差幀通道能提供原始影像缺乏的移動線索，使模型更容易從複雜背景中辨識正在移動的微小垃圾，明顯降低漏判情形，驗證時間資訊對低幀率監視影像中微小物件偵測的有效性。

[P00111 | 3234:3235 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00112 | 3235:3236 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00113 | 3236:3237 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00114 | 3237:3255 | HEADING_2]
YOLO-Segmentation

[P00115 | 3255:3542 | NORMAL_TEXT]
開發初期，我們採取演進式架構（Evolutionary Architecture）的開發方式，先迅速搭建系統的骨架，後續再逐步調整與優化各模組的實作細節。其中，關於人、車與垃圾的物件辨識模型，我們統一採用 YOLO26 Detection 的 large 模型，基於資料集(一)進行訓練。在優化階段時，我們為了篩除 YOLO26-L 會誤將車子部件（如車燈、後照鏡）辨識為垃圾的問題，制定了一系列的後處理機制。但後來，我們發現在部分案例中，存在「垃圾自丟棄至落地，始終落在車子 bbox 中，導致被後處理篩除」的問題。因此我們開始思考，「該如何準確分割車子的邊界與背景?」

[P00116 | 3542:3807 | NORMAL_TEXT]
YOLO-Segmentation 在物件偵測之外，額外輸出每個實例的像素級遮罩（instance mask），讓系統同時取得類別、信心分數、bbox 與可見輪廓。相較於矩形 bbox，遮罩能更細緻地描述車身形狀，將矩形框內的道路與背景區域區分出來。表三列出 [Ultralytics 官方](https://docs.ultralytics.com/models/yolo26/) COCO val2017、640 像素輸入及端到端（e2e）設定下的比較；各尺度的 Box mAP50–95 差異為 0 至 −0.5 個百分點，而 Mask mAP50–95 是另一項分割指標，不能直接當成本專題的垃圾辨識或歸因準確率。

[P00117 | 3807:3982 | NORMAL_TEXT]
分割亦增加推論成本。以官方 large 模型的 T4 TensorRT10 數據為例，Detection 為 6.2 ms，[Segmentation](https://docs.ultralytics.com/tasks/segment/) 為 8.0 ms，延遲約增加 29%。這僅是相同官方基準下的單模型比較，不能直接推算本系統的完整影片處理速度。本研究選擇分割模型，是希望以額外的輪廓資訊改善僅用 bbox 判斷車體與垃圾關係的限制。

[P00118 | 3985:3991 | NORMAL_TEXT | TABLE row=0 col=0]
Model

[P00119 | 3992:4019 | NORMAL_TEXT | TABLE row=0 col=1]
YOLO26 Detect Box mAP50-95

[P00120 | 4020:4044 | NORMAL_TEXT | TABLE row=0 col=2]
YOLO26-Seg Box mAP50-95

[P00121 | 4045:4048 | NORMAL_TEXT | TABLE row=0 col=3]
差異

[P00122 | 4049:4067 | NORMAL_TEXT | TABLE row=0 col=4]
Seg Mask mAP50-95

[P00123 | 4069:4071 | NORMAL_TEXT | TABLE row=1 col=0]
n

[P00124 | 4072:4077 | NORMAL_TEXT | TABLE row=1 col=1]
40.1

[P00125 | 4078:4083 | NORMAL_TEXT | TABLE row=1 col=2]
39.6

[P00126 | 4084:4089 | NORMAL_TEXT | TABLE row=1 col=3]
-0.5

[P00127 | 4090:4095 | NORMAL_TEXT | TABLE row=1 col=4]
33.9

[P00128 | 4097:4099 | NORMAL_TEXT | TABLE row=2 col=0]
s

[P00129 | 4100:4105 | NORMAL_TEXT | TABLE row=2 col=1]
47.8

[P00130 | 4106:4111 | NORMAL_TEXT | TABLE row=2 col=2]
47.3

[P00131 | 4112:4117 | NORMAL_TEXT | TABLE row=2 col=3]
-0.5

[P00132 | 4118:4123 | NORMAL_TEXT | TABLE row=2 col=4]
40.0

[P00133 | 4125:4127 | NORMAL_TEXT | TABLE row=3 col=0]
m

[P00134 | 4128:4133 | NORMAL_TEXT | TABLE row=3 col=1]
52.5

[P00135 | 4134:4139 | NORMAL_TEXT | TABLE row=3 col=2]
52.5

[P00136 | 4140:4144 | NORMAL_TEXT | TABLE row=3 col=3]
0.0

[P00137 | 4145:4150 | NORMAL_TEXT | TABLE row=3 col=4]
44.1

[P00138 | 4152:4154 | NORMAL_TEXT | TABLE row=4 col=0]
l

[P00139 | 4155:4160 | NORMAL_TEXT | TABLE row=4 col=1]
54.4

[P00140 | 4161:4166 | NORMAL_TEXT | TABLE row=4 col=2]
54.4

[P00141 | 4167:4171 | NORMAL_TEXT | TABLE row=4 col=3]
0.0

[P00142 | 4172:4177 | NORMAL_TEXT | TABLE row=4 col=4]
45.5

[P00143 | 4179:4181 | NORMAL_TEXT | TABLE row=5 col=0]
x

[P00144 | 4182:4187 | NORMAL_TEXT | TABLE row=5 col=1]
56.9

[P00145 | 4188:4193 | NORMAL_TEXT | TABLE row=5 col=2]
56.5

[P00146 | 4194:4199 | NORMAL_TEXT | TABLE row=5 col=3]
-0.4

[P00147 | 4200:4205 | NORMAL_TEXT | TABLE row=5 col=4]
47.0

[P00148 | 4206:4272 | NORMAL_TEXT]
表三、YOLO26 與 YOLO26-Seg accuracy 在不同模型的差異性（資料來源：Ultralytics 官方文件）

[P00149 | 4272:4273 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00150 | 4273:4375 | NORMAL_TEXT]
在目前系統中，YOLO-Seg 負責車輛與機車的偵測、追蹤及輪廓擷取；人物與骨架由 YOLO-Pose 提供，垃圾候選則由四通道 RT-DETR 產生。各模型分工明確，車輛遮罩本身不會直接確認垃圾事件。

[P00151 | 4375:4598 | NORMAL_TEXT]
遮罩主要用於判斷垃圾候選是否仍附著於車體。後處理計算垃圾 bbox 被車輛遮罩覆蓋的比例，以及垃圾位置到輪廓的有號距離，再結合跨幀位移、相對運動與脫離車身的證據，區分車燈、後照鏡等車體部件與可能正在落下的垃圾。例如，垃圾雖仍位於車輛的大矩形框內，卻已離開可見車體輪廓，便不應只因 bbox 重疊而立即排除；但它仍須通過其餘運動與事件確認條件，不能僅憑離開遮罩就認定丟棄成立。缺少有效 mask 時，系統退回 bbox 與相對運動等既有條件處理。

[P00152 | 4598:4720 | NORMAL_TEXT]
此外，系統保存部分真實觀測輪廓，供 Backtrack 比較垃圾釋放前後與車體的相對位置。可見遮罩會受到遮擋與分割誤差影響，因此「垃圾位於某車 mask 內」只表示影像平面上的重疊，不代表該車就是來源；歸因仍須整合人物、車輛與垃圾的時間序列。

[P00153 | 4720:4876 | NORMAL_TEXT]
運算方面，YOLO-Seg 可透過跳幀與快取降低重複推論，並以最近一次真實車輛觀測維持車輛閘門。預設在最近 3 秒內有車輛觀測時啟用後續分析；逾時可略過姿態、動作與垃圾偵測等昂貴路徑。快取不會刷新此時間窗。此設計適合以車輛關聯為重點的場景，但也表示車輛漏偵測或純行人場景可能使後續分析被略過，需依部署需求調整。

[P00154 | 4876:4910 | NORMAL_TEXT]
(要補 yolo-seg 與 yolo bbox 實際案例對照圖)

[P00155 | 4910:4911 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00156 | 4911:4912 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00157 | 4912:4932 | HEADING_2]
YOLO-Pose & STGCN++

[P00158 | 4932:5130 | NORMAL_TEXT]
在已知系統必須辨識「隨地便溺」行為的情況下，我們起初也希望利用同一套動作辨識模型判斷「丟棄垃圾」，藉此彌補部分垃圾因體積過小、受到遮擋，或受監視器解析度與拍攝距離限制，而無法被物件偵測模型穩定辨識的問題。因此，初期採用 YOLO-Pose 搭配 STGCN++，將動作分為 {0: normal, 1: urinate, 2: littering} 三類，嘗試從人物動作補足垃圾物件資訊的不足。

[P00159 | 5130:5417 | NORMAL_TEXT]
在模型選擇上，YOLO-Pose 除了取得人物的 bounding box，還能輸出 COCO 格式的 17 個人體關鍵點及其信心分數，包括肩膀、手肘、手腕、髖部、膝蓋與腳踝等位置。每個關鍵點可用（x, y, confidence）表示，將影像轉換成較精簡的人體姿態資料，讓後續分類器著重關節的相對位置與移動，而不直接依賴衣服顏色或背景紋理。現行系統使用 YOLO-Pose 的追蹤流程維持人物 ID，並直接將同一人物的 bbox 與 keypoints 對齊，無須另外將另一個偵測模型的人物框與骨架進行 IoU 配對。（參考：[Ultralytics Pose 官方文件](https://docs.ultralytics.com/tasks/pose/)）

[P00160 | 5417:5662 | NORMAL_TEXT]
不過，單一影格的姿態不足以描述完整動作，因此我們進一步採用 STGCN++ 分析骨架序列。模型將人體視為時空圖：關節是節點，肩膀與手肘等人體連接構成空間關係，連續影格中的關節變化則提供時間資訊。空間圖卷積學習不同關節的相互關係，時間卷積分析姿態的變化與持續模式。本研究採用的設定包含可學習的圖連接權重、多尺度時間卷積及殘差連接，使模型能整合不同時間尺度的動作線索。兩者的組合形成「影像→人體骨架→時序分類」的流程，將姿態擷取與行為判斷分工處理。（參考：[MMAction2 STGCN++](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/stgcnpp/README.md)）

[P00161 | 5662:5878 | NORMAL_TEXT]
然而，初期三分類模型實際套用至監視器影片時，經常將 urinate 與 littering 混淆。與教授討論後，我們認為可能原因是兩類行為在二維骨架特徵上的差異不足：站立、身體略微前傾、手部靠近腰部等姿態，可能同時出現在不同動作中；真正區分丟垃圾的「物體是否離手」，則不包含在人體關節座標裡。拍攝角度、遮擋與關鍵點誤差，也可能進一步掩蓋短暫的手部動作。這是我們根據實際案例形成的解釋，尚不能單憑類別混淆就斷定模型內部特徵完全相同。

[P00162 | 5878:6075 | NORMAL_TEXT]
基於上述觀察，我們後續將 STGCN++ 縮減為 {0: normal, 1: urinate} 二分類，專注辨識隨地便溺；亂丟垃圾則改由 RT-DETR 的垃圾候選、移動證據、跨幀事件確認與 Backtrack 負責。如此一來，即使人物丟垃圾時的骨架被分類為 normal，只要垃圾物件事件的證據充分，仍可成立 littering 事件；反之，也不會只因人物姿勢相似，就推定畫面中存在垃圾。

[P00163 | 6075:6371 | NORMAL_TEXT]
在訓練資料蒐集階段，我們從嘉義縣環保局提供的 141 部隨地便溺影片中，針對行為發生區間切割約 8～10 秒的片段，初期整理出 265 部正例 clips；另外加入走路、手握方向盤、站立及滑手機等正常行為，共整理 164 部、長度約 4～9 秒的負例 clips，讓模型學習違規姿態與常見正常動作的差異。上述數量是初期影片整理紀錄，不能直接視為最後送入模型的樣本數。現存骨架資料版本共有 840 筆 annotations，其中 normal 614 筆、urinate 226 筆，分為訓練集 671 筆與驗證集 169 筆；這些 annotation 數量亦不等於彼此獨立的原始影片數。

[P00164 | 6371:6637 | NORMAL_TEXT]
影片先由 YOLO-Pose 轉成骨架序列，再經座標正規化與時間取樣輸入 STGCN++。訓練設定從片段取樣 100 個骨架影格；線上推論則依人物 ID 累積歷史，達到 100 個有效骨架後開始分類。為減少人物大小與位置不同造成的分布差異，推論時以可信關鍵點的外接範圍平移、調整尺度，並對部分低信心關節做時間補值與平滑。現行輸入保留 17 點格式，將鼻、眼、耳等前五個頭部節點的信心設為零，以對齊目前訓練資料；這並非另建一套 12 點骨架模型。此外，100 個影格涵蓋的時間會隨影片 FPS 與取樣跨度改變，不能一律視為固定秒數。

[P00165 | 6637:6899 | NORMAL_TEXT]
成果方面，目前使用的 best_stgcn_0623.pth 在第 11 個 epoch 保存的驗證紀錄中，Top-1 Accuracy 為 93.49%，對應 169 筆驗證 annotations 中 158 筆分類正確，Mean Class Accuracy 亦約為 93.49%。驗證集包含 normal 123 筆與 urinate 46 筆。這是既有資料切分下的片段級驗證成績，不是完整影片事件偵測、車輛歸因或自動開罰的準確率；二分類與早期三分類的任務及資料條件不同，也不能直接把分數差異全歸因於移除一個類別。

[P00166 | 6899:7149 | NORMAL_TEXT]
在實際運行時，系統不會因一次分類為 urinate 就立即建立事件，而是再累積時間證據。現行預設在最近 8 秒內，將 urinate 分數不低於 0.2 的逐幀貢獻除以 FPS 後加總，達到 3.5 分數·秒，且當下分類仍為 urinate，才觸發確認。這項累積值不是「已確定便溺的秒數」，而是用來整合持續性分類證據；另保留以高低門檻累積正例時間的替代模式。此設計旨在減少短暫姿態與分類波動直接形成事件，但遮擋、人物追蹤中斷及正常動作相似等問題仍須透過獨立場域與完整影片評估，最終結果保留人工複核。

[P00167 | 7149:7150 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00168 | 7150:7151 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00169 | 7151:7167 | HEADING_2]
反追蹤模組 Backtrack

[P00170 | 7167:7350 | NORMAL_TEXT]
反追蹤模組的目的，是在垃圾事件成立後，回溯垃圾可能由哪一位人物或哪一輛車釋放，建立「事件—人物—車輛」的證據關係。垃圾首次被 RT-DETR 偵測的影格，通常不等於實際釋放影格；若直接選擇確認當下最近的車輛，多車交會、遮擋與偵測延遲便可能造成錯誤歸因。因此，本研究結合跨幀身分銜接、軌跡平滑、釋放假設與路徑成本，評估多種可能來源，並保留無法判定的 NULL 結果。

[P00171 | 7350:7351 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00172 | 7351:7364 | NORMAL_TEXT]
（一）跨幀身分與軌跡整理

[P00173 | 7364:7659 | NORMAL_TEXT]
系統先保存 YOLO-Seg 的車輛／機車，以及 YOLO-Pose 的人物歷史觀測，包括 bbox、track ID、影格編號與偵測信心。匈牙利演算法（Hungarian Algorithm）搭配運動預測，用於同類物件在相鄰影格間的一對一身分銜接，以降低 ID switch；它不負責人物與車輛之間的配對，也不直接判斷垃圾來源。接著以信心感知的 Kalman filter 估計位置與速度，再以 Rauch–Tung–Striebel（RTS）smoothing 利用歷史序列的前後資訊減少定位抖動。遮擋期間的預測點與跳幀快取均不視為新的偵測證據，並保留最近真實觀測時間及位置不確定性。

[P00174 | 7659:7660 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00175 | 7660:7675 | NORMAL_TEXT]
（二）垃圾釋放時間與位置假設

[P00176 | 7675:7961 | NORMAL_TEXT]
只有通過 GlobalLitterTracker 確認的垃圾事件，才會建立反追蹤工作並交由背景執行緒處理；尚未確認的 candidate 不會因為附近有人車而直接成為違規事件。模組取垃圾早期飛行軌跡，依影片 FPS 將影格換算為秒，在影像座標中擬合水平一次、垂直二次的運動模型，向前回推可能的釋放位置與時間。這是短時間的影像軌跡近似，並非已校正的三維拋體重建。僅有兩個有效觀測時，改採常速假設並提高不確定性；只有單點時則施加保守成本，避免缺乏軌跡仍強行判定來源。每個釋放假設都與同一時刻的人車位置對齊，觀測過舊的假設會受到相應成本限制，回推範圍則受設定的搜尋時間窗約束。

[P00177 | 7961:7962 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00178 | 7962:7977 | NORMAL_TEXT]
（三）三類關聯成本與完整路徑

[P00179 | 7977:8201 | NORMAL_TEXT]
令 B 為垃圾、A 為人物、C 為車輛，分別建立 C_BA（垃圾→人物）、C_AC（人物→車輛）與 C_BC（垃圾→車輛）成本。C_BA 評估釋放位置與人物上半身釋放區域的距離、觀測品質及不確定性；C_AC 依人物與車輛的持續接近、軌跡端點及進出車輛線索建立關聯，不能只因兩個 bbox 重疊就配對；C_BC 評估垃圾釋放位置與車輛區域的一致性。距離以人物或車輛尺度正規化，並保留獨立的幾何限制，避免放大預測不確定性後反而讓遠處物件成為合理候選。

[P00180 | 8201:8339 | NORMAL_TEXT]
上述成本組成「垃圾→人物→車輛」、「垃圾→人物→未知車輛」、「垃圾→車輛」及「垃圾→NULL」四類路徑。人物路徑中的車輛可以透過先前的人車關係找到，因此不要求人物丟棄時仍站在車旁；直接車輛路徑則必須通過垃圾與該車之間的幾何條件。系統比較完整路徑，而非只保留離垃圾最近的人物。

[P00181 | 8339:8340 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00182 | 8340:8356 | NORMAL_TEXT]
（四）最小成本流與遮罩時序檢查

[P00183 | 8356:8494 | NORMAL_TEXT]
系統以最小成本流（Min-Cost Flow）為每個事件選出一條路徑。事件各自展開候選節點，不對同一人物或車輛施加全域的一對一容量限制，因此可表示同車多人與同人多次丟棄。每個事件始終保留完整 NULL 路徑，當有效候選不足或其成本不如 NULL 時，保留事件並輸出未知歸因。

[P00184 | 8494:8675 | NORMAL_TEXT]
初步選取後，模組再以受限制的路徑比較檢查既有有效候選，其中包括 YOLO-Seg 的跨幀遮罩線索：觀察釋放前後垃圾位置相對可見車體輪廓的內外變化，並結合軌跡模型、幾何距離及不確定性判斷是否重選。此步驟只使用真實觀測的遮罩；缺少必要輪廓或證據時維持原選擇，不改寫垃圾確認結果，也不放寬候選的幾何限制。單幀 mask 重疊既不代表深度，也不能直接證明垃圾來自該車。

[P00185 | 8675:8676 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00186 | 8676:8687 | NORMAL_TEXT]
（五）輸出與評估方式

[P00187 | 8687:8862 | NORMAL_TEXT]
輸出保留人物／車輛 ID 或 NULL、釋放假設、路徑成本與候選差距，供人工查核。成本差距（margin）反映候選之間的區別程度，並非準確率或校準後機率；目前實作將其保存為診斷資訊，未以單一固定 margin 門檻自動改判 NULL。找到可靠關聯車輛後，才將原始未畫框影像中的車輛區域交給車牌模組；若 OCR 失敗，保留失敗狀態與證據，不補造車牌。

[P00188 | 8862:9089 | NORMAL_TEXT]
本研究以已整理的 58 部困難垃圾案例作為開發與比較資料，評估時需分開統計事件是否被確認，以及確認事件的人物、車輛與釋放位置是否符合人工標註；模型 track ID 必須配合該次執行的 bbox 與影格核對，不能將不同執行的相同數字 ID 視為同一物件。候選紀錄與自動輸出的 sidecar 只是診斷資料，不能取代人工 ground truth。同一批開發案例上的調參結果亦不能代表跨攝影機的泛化能力；正式成效仍需以獨立場域及人工複核的正、負案例驗證。

[P00189 | 9089:9091 | NORMAL_TEXT]
[INLINE_OBJECT kix.q41iwt1nt7w0]

[P00190 | 9091:9115 | NORMAL_TEXT]
圖四、Backtrack 反追蹤模組運作流程圖

[P00191 | 9115:9116 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00192 | 9116:9117 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00193 | 9117:9118 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00194 | 9118:9119 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00195 | 9119:9120 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00196 | 9120:9133 | HEADING_2]
Optimization

[P00197 | 9133:9163 | NORMAL_TEXT]
RT-DETR P2 Feature Extraction

[P00198 | 9166:9177 | NORMAL_TEXT | TABLE row=0 col=0]
Model size

[P00199 | 9178:9188 | NORMAL_TEXT | TABLE row=0 col=1]
Percision

[P00200 | 9189:9197 | NORMAL_TEXT | TABLE row=0 col=2]
Recall 

[P00201 | 9198:9204 | NORMAL_TEXT | TABLE row=0 col=3]
mAP50

[P00202 | 9205:9214 | NORMAL_TEXT | TABLE row=0 col=4]
mAP50-95

[P00203 | 9216:9226 | NORMAL_TEXT | TABLE row=1 col=0]
1280×1280

[P00204 | 9227:9232 | NORMAL_TEXT | TABLE row=1 col=1]
0.78

[P00205 | 9233:9238 | NORMAL_TEXT | TABLE row=1 col=2]
0.63

[P00206 | 9239:9244 | NORMAL_TEXT | TABLE row=1 col=3]
0.63

[P00207 | 9245:9250 | NORMAL_TEXT | TABLE row=1 col=4]
0.32

[P00208 | 9252:9260 | NORMAL_TEXT | TABLE row=2 col=0]
980×980

[P00209 | 9261:9266 | NORMAL_TEXT | TABLE row=2 col=1]
0.69

[P00210 | 9267:9272 | NORMAL_TEXT | TABLE row=2 col=2]
0.64

[P00211 | 9273:9278 | NORMAL_TEXT | TABLE row=2 col=3]
0.66

[P00212 | 9279:9283 | NORMAL_TEXT | TABLE row=2 col=4]
0.3

[P00213 | 9285:9298 | NORMAL_TEXT | TABLE row=3 col=0]
980×980 + P2

[P00214 | 9299:9304 | NORMAL_TEXT | TABLE row=3 col=1]
0.76

[P00215 | 9305:9310 | NORMAL_TEXT | TABLE row=3 col=2]
0.58

[P00216 | 9311:9316 | NORMAL_TEXT | TABLE row=3 col=3]
0.64

[P00217 | 9317:9322 | NORMAL_TEXT | TABLE row=3 col=4]
0.34

[P00218 | 9323:9358 | NORMAL_TEXT]
表四、RT-DETR 在不同輸入尺寸及有無 P2 特徵層下的效能比較

[P00219 | 9358:9729 | NORMAL_TEXT]
	我們參考〈A Simple Detector with Frame Dynamics is a Strong Tracker〉[3]中利用 P2 高解析度特徵層保留細粒度資訊的方法，將 P2 特徵層加入模型，以提升微小垃圾的偵測能力。實驗結果顯示，在輸入尺寸為 980×980 時，加入 P2 特徵層後，Precision 由 0.69 提升至 0.76，mAP50–95 由 0.30 提升至 0.34；與1280×1280 的模型相比，其 Precision 僅相差 0.02，mAP50 與 mAP50–90 也有些微的提升。雖然 Recall 略微下降，但整體偵測表現已接近 1280×1280 的模型。這表示 P2 特徵層能在較小輸入尺寸下保留更多微小物件的細節，使模型在降低約 41% 輸入像素量的同時，仍維持良好的偵測效果。

[P00220 | 9729:9730 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00221 | 9730:9740 | NORMAL_TEXT]
Pipeline 

[P00222 | 9740:9849 | NORMAL_TEXT]
在程式執行過程中，我們發現 GPU 使用率大多僅約 25%，未能充分發揮運算效能。經初步分析，效能瓶頸出現在資料前處理階段(計算Pixel Change Map)，使 GPU 必須等待資料輸入，進而影響整體運算效率。

[P00223 | 9849:9851 | NORMAL_TEXT]
[INLINE_OBJECT kix.2em3n4m9jp4k]

[P00224 | 9851:9881 | NORMAL_TEXT]
圖五、Pipeline架構（上）及執行流程（下）簡易示意圖

[P00225 | 9881:10083 | NORMAL_TEXT]
為了解決上述的問題，本專案採用類似 CPU 指令 pipelining 的設計如圖O所示，讓不同批次的影格同時位於不同處理階段，以提高單位時間內可完成的影格數量。 每張影格仍須依序經過讀取、前處理、模型預測、後處理及輸出，但系統透過背景執行緒與queue，使不同批次的工作能夠重疊執行。此外，RT-DETR 採用batch size = 8 的方式輸入影像，以提高模型的整體推論 throughput。

[P00226 | 10083:10084 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00227 | 10084:10093 | HEADING_1]
（四）成果與總結

[P00228 | 10093:10198 | NORMAL_TEXT | LIST id=kix.xe7gbtxnohjr level=0]
採用四通道 RT-DETR，融合 RGB 影像與 Pixel Change Map，使 Recall 與 mAP50 提升至原模型的約兩倍，並成功辨識出許多原模型難以偵測、尺寸微小或容易與背景混淆的垃圾物件。

[P00229 | 10198:10253 | NORMAL_TEXT | LIST id=kix.xe7gbtxnohjr level=0]
加入 P2 高解析度特徵層，使模型在降低輸入尺寸與運算量的同時，仍能維持良好的辨識準確度，進而提升推論速度。

[P00230 | 10253:10339 | NORMAL_TEXT | LIST id=kix.xe7gbtxnohjr level=0]
透過推論 pipeline 與批次處理最佳化，在尚未使用 TensorRT 加速的情況下，將系統吞吐量由 7.66 FPS 提升至 16 FPS，約為原先的 2.1 倍。

[P00231 | 10339:10340 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00232 | 10340:10348 | HEADING_1]
（五）引用文獻

[P00233 | 10348:10349 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00234 | 10349:10605 | NORMAL_TEXT]
[1] Corsel, C. W., Van Lier, M., Kampmeijer, L., Boehrer, N., & Bakker, E. M. (2023, January). Exploiting temporal context for tiny object detection. In 2023 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW) (pp. 1-11). IEEE.

[P00235 | 10605:10820 | NORMAL_TEXT]
[2] Nans, L., Mediavilla, C., Marez, D., & Parameswaran, S. (2023). Leveraging motion saliency via frame differencing for enhanced object detection in videos. Pattern Recognition and Tracking XXXIV, 12527, 125270V.

[P00236 | 10820:11078 | NORMAL_TEXT]
[3] Peng, C., Wang, C., Zou, M., Li, D., Yang, Z., Dai, Y., ... & Li, X. (2025, June). A simple detector with frame dynamics is a strong tracker. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) (pp. 6632-6642). IEEE.

