# 期中考參考答案

## 1. 請比較基於預訓練語料的詞向量 (例如Word2Vec) 與語言模型詞向量 (如RNN、ELMo、BERT) 的差異。
Word2Vec 的詞向量是 static embeddings，每個詞的向量在不同的上下文皆為固定。
語言模型詞向量 (如RNN、ELMo、BERT) 屬於 contextual embeddings，每個詞的向量會依據上下文而變動。

## 2. 請說明什麼是 Language models 以及 (常見的) Language models 的訓練方式 (Language modeling)。
Language models 是預測目標字詞 (或 token) 並給予機率值或機率分佈的模型。
常見的 Language modeling 有 Next-token prediction、Masked Language Modeling 等。

## 3. Sub-word Tokenization 主要可以解決什麼問題？
Out-of-vocabulary 的問題。

## 4. 為什麼 Self-attention 的過程要有 Queries, Keys, 和 Values？
Queries, Keys, 和 Values 的存在是為了讓模型去搜尋 (Queries) 序列中的內容 (Keys)，並計算輸入序列中彼此之間的重要性 (attention)，進而產生具備上下文關聯性的表達 (Values)。

## 5. 為什麼 Transformers 需要加上Positional Encodings？為什麼RNN不用？
Transformers 的 self-attention 計算方法並沒有考慮的輸入序列的順序性，因此需要額外的 Positional Encodings。
RNN 的作法是遞迴處理輸入序列每個時間點的內容，本身所產生的 hidden states (contextual embeddings) 已具備順序性，因此不需要額外的 Positional Encodings。

## 6. (b)(c)(d)

## 7. 程式觀念題1
(1) (2, 3, 2)
(2) broadcasting
(3) (2, 3, 2)

## 8. 程式觀念題2
torch.nn.CrossEntropyLoss() 已經包含 softmax 計算，因此模型的輸出不需要先經過 torch.nn.Softmax

## 9. 程式觀念題3
tensor([[1, 1],
        [1, 1]])

## 10. 程式觀念題4
(1) (B, 100, 300)
(2) (B, 100, 300)
(3) (B, 100, 600)