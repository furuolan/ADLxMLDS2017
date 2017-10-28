# HW1

檔案說明：
1. model_xxx.json 為 model 的 structure

2. model_xxx.hdf5 為 model 的 weight

3. model_predict.py 為 predict test dataex. 
    > python model_predict.py model_best data/ result.cs
 
4. hw1_xxx.sh 為執行 model_predict.py，其中已預設要使用的 model name

5. model_xxx.py 為 training code，
    > python model_best.py data/ bestmodel
    >
    bestmodel為要儲存的 model name～
6. 40class.pkl 為phones 之 one hot encoding

