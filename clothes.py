import streamlit as st
import xgboost as xgb
import pandas as pd

# 服装指数マッピング辞書の作成
clothes_mapping = {
    1: '冬物コート',
    2: 'トレンチコート',
    3: 'セーター',
    4: 'カーディガン',
    5: '長袖',
    6: '半袖',
    7: 'ダウンコート'
}

# 事前にトレーニングしたモデルをロード (パスは適切なものに変更してください)
bst = xgb.Booster({'n_jobs': 4})
bst.load_model('model.bst')

st.title('服装指数予測アプリ')

# ユーザーに気温を入力させる
temperature = st.number_input('本日の気温を入力してください:', min_value=0, max_value=40)

if st.button('予測する'):
    # 入力データを作成
    input_data = pd.DataFrame([temperature], columns=['気温'])
    
    # XGBoost用の型に変換
    dtest = xgb.DMatrix(input_data)
    
    # 予測
    pred = bst.predict(dtest)
    
    # 予測結果を服装指数にマッピング
    pred_clothes = clothes_mapping[int(pred[0])]
    
    # 結果を表示
    st.write(f'おすすめの服装は: {pred_clothes}')
