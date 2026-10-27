# -
MINIST手寫辨識
# AI Projects｜利用神經網路建立 MNIST 手寫數字辨識深度學習模型


這個專案使用 TensorFlow / Keras 框架，對 MNIST 手寫數字數據集進行圖像辨識的深度學習模型設計、訓練與評估。專案由數據載入開始，依序完成模型建構、編譯、訓練與測試集評估，最後將測試圖像視覺化，讓初學者能夠完整走過一次深度學習的基本流程，並理解神經網路如何將 28×28 的灰階影像對應到 0～9 的數字類別。

---

## 1. 學習目標

- 瞭解深度學習（Deep Learning）和神經網路（Neural Networks）的基本概念。
- 掌握使用 TensorFlow 進行數據載入、模型建構、訓練和評估的技巧。
- 認識 ReLU 與 Softmax 激活函數在分類任務中的角色。
- 實作建構、訓練和評估一個多層感知器（MLP）分類模型。
- 學會解讀訓練過程中的 accuracy 與 loss，並提升問題解決和模型優化的能力。

---

## 2. 步驟說明

### Step 1. 數據載入

使用 TensorFlow 的 Keras API 載入 MNIST 數據集，這是最常提供給初學者使用的圖像分類數據集，由 60,000 張訓練圖像（training set）和 10,000 張測試圖像（test set）組成。每張圖像都是 28×28 的灰階影像，並對應 0～9 其中一個數字標籤。

```python
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)
```

---

### Step 2. 定義模型結構

使用 `Sequential` 模型將各層依序堆疊。首先是平坦層（Flatten Layer），將 28×28 的二維圖像轉換為 784 維的一維向量；接著是兩個密集層（Dense Layer），分別負責特徵學習與最後的分類輸出：

- `Dense(128, activation="relu")`：128 個神經元，使用 ReLU 激活函數學習特徵。
- `Dense(10, activation="softmax")`：10 個神經元，對應數字 0～9，Softmax 會輸出每個類別的機率。

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

各層參數數量如下：

| 層 | 輸出形狀 | 參數數量 | 計算方式 |
|---|---|---|---|
| Flatten | 784 | 0 | 只做數據轉換，沒有可訓練參數 |
| Dense (128) | 128 | 100,480 | 784 × 128 + 128（bias） |
| Dense (10) | 10 | 1,290 | 128 × 10 + 10（bias） |
| **合計** | | **101,770** | 全部皆為可訓練參數 |

---

### Step 3. 模型編譯與訓練

**編譯模型**：訓練之前需要設定優化器、損失函數和評估指標。本專案使用 Adam 優化器，以 `sparse_categorical_crossentropy` 作為損失函數（標籤為整數 0～9，不需要先轉成 one-hot），並追蹤準確率（accuracy）作為評估指標。

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

**訓練模型**：使用 `fit` 方法以訓練數據訓練模型，過程中模型會學習如何將輸入圖像映射到輸出類別，這就是監督式學習的基本概念。這裡設定迭代 5 個訓練週期（epochs），讀者可以依需求調整並觀察訓練效果。

```python
model.fit(x_train, y_train, epochs=5)
```

訓練過程紀錄：

| Epoch | Accuracy | Loss |
|---|---|---|
| 1 | 0.8610 | 2.7848 |
| 2 | 0.9094 | 0.3847 |
| 3 | 0.9276 | 0.2857 |
| 4 | 0.9369 | 0.2460 |
| 5 | 0.9425 | 0.2268 |

可以看到準確率逐步上升、損失逐步下降，表示模型正在有效學習。

---

### Step 4. 模型評估

使用從未參與訓練的測試數據集來評估模型，以了解模型面對沒看過的數據時的表現，也就是泛化能力（Generalization）。

```python
loss, accuracy = model.evaluate(x_test, y_test)
print("測試準確率：", round(accuracy * 100, 2), "%")
```

執行結果：

```
測試準確率： 94.0 %
```

測試損失約為 0.2720，測試準確率為 94.0%，表示模型在 10,000 張測試圖像中，約有九成四能夠正確辨識。

---

### Step 5. 顯示測試圖像

為了直觀確認資料內容，將測試集的第一張圖像顯示出來，並在標題附上它的實際數字標籤。

```python
plt.imshow(x_test[0], cmap="gray")
plt.title("Actual digit: " + str(y_test[0]))
plt.axis("off")
plt.show()
```

> 圖片位置：貼上程式輸出的測試圖像（標題為 `Actual digit: 7`）

---
### Step 6.成果展示

![執行結果](-/螢幕擷取畫面%202026-10-27%20192200.png)

---


### Step 7. 優化方向（延伸練習）

本專案的基本模型已經能達到 94% 的準確率，若想進一步提升，可以嘗試以下調整，並觀察對結果的影響：

- **數據正規化**：將像素值除以 255，縮放到 0～1 之間，通常能讓訓練更穩定，也能降低一開始過大的 loss（本專案第 1 個 epoch 的 loss 為 2.78）。

  ```python
  x_train = x_train / 255.0
  x_test = x_test / 255.0
  ```

- **調整訓練週期**：增加 `epochs` 次數，觀察準確率是否繼續提升或出現過擬合。
- **調整模型結構**：增加 Dense 層數或改變神經元數量。
- **使用卷積神經網路（CNN）**：對圖像任務通常有更好的表現。

---

## 3. 心得與結論

透過這個專案，我們完整走過了深度學習的基本流程：載入數據、建立神經網路、編譯與訓練模型，再到測試集評估與結果視覺化。僅僅使用一個隱藏層的簡單全連接網路，在訓練 5 個 epoch 後就能在測試集上達到 94.0% 的準確率，顯示即使是基礎的神經網路，也足以處理手寫數字辨識這類問題。

同時也觀察到，第一個 epoch 的 loss 偏高，這與輸入數據沒有做正規化有關，讓我體會到數據預處理對模型訓練的重要性。未來可以嘗試正規化、調整模型參數與層數，甚至改用 CNN，來進一步提升模型表現。

---

## 執行環境

- Python 3
- TensorFlow / Keras
- Matplotlib
- 建議使用 Google Colab 執行，無需額外安裝環境

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `人工智慧期中作業.ipynb` | 完整程式碼與執行結果 |
| `README.md` | 專案說明文件 |

## 參考資料

- MNIST 數據集：https://keras.io/api/datasets/mnist/
