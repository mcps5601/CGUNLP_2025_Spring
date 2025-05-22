# Kaggle submission hints
- 以下是針對 NBME - Score Clinical Patient Notes 做的一些測試結果
- 但是其他競賽有些部分也是適用的
- 歡迎私訊傳 Code 給老師，我可以幫 Debug



============



假設各位已經正在自己的競賽 Notebook 進行 Code 編輯



## 1. 離線版本的套件取得
- 步驟1: 在 Kaggle 開一個Notebook 或Dataset (左上角+號)
- 步驟2: 新增 Code，可參考 https://www.kaggle.com/code/mcps5601/notebook41a63761c3
- 步驟3: 掛載到自己的競賽 Notebook 中
- 步驟4: 指定安裝位置，進行離線版本套件的安裝
    - 可在 Notebook 中使用以下指令來查詢掛載的內容
    ```
    !ls ../input/
    ```

## 2. Debug for Scoring

當你的 Code 完成後，Kaggle 會：

(1) 先用你的 Code 跑一次 sample_submission

(2) 才會真正開始跑 hidden test set 的預測，也就是 Scoring



第(2)步可能會出現 Notebook Threw Exception

但因為沒有 log，所以難以 Debug



Kaggle Scoring 是基於 hidden test set，但這通常相當巨大 (所以會跑超過30分鐘)

然而，sample_submission 採用 test.csv，數量太少不足以顯示所有資料都不會有問題

因此各位可以在 Kaggle 上測試 Notebook 時，將 test 的變數設為 train.csv

(請參閱: https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/318215)

這樣你可以有足夠多的資料進行測試，有機會能夠幫助 Debug



## 3. NBME code 範例

https://www.kaggle.com/code/mcps5601/notebookc5a9c9ffd8

以上連結是我基於 [yasufuminakama 的 code](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-inference) 改寫的版本 (我只有使用第0個fold，有fine-tuning)

yasufuminakama 使用的 PyTorch 版本老舊 (1.9.x)，不符合目前 Kaggle 的執行環境

我使用的版本是 torch==2.4.0 和 transformers==4.41.0，經測試沒有問題