import numpy as np
import pandas as pd
from string import printable
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import sequence
from keras.models import load_model


def create_scaler(df):
    cols_to_scale = ['html_length', 'n_hyperlinks', 'n_script_tag', 'n_link_tag', 'n_comment_tag']
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[[c + '_std' for c in cols_to_scale]] = scaler.fit_transform(df_scaled[cols_to_scale].astype(float))
    df_scaled = df_scaled.drop(columns=cols_to_scale)
    return df_scaled


def create_X_1(temp_X_1):
    url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in temp_X_1.url]
    max_len = 150
    X_new_1 = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
    return X_new_1


def create_X_2(temp_X_2):
    x = temp_X_2.drop(columns=['url']).values.astype(float)
    X_new_2 = x.reshape(x.shape[0], x.shape[1], 1)
    return X_new_2


def main(n_samples: int = 20):
    legitimate_test = pd.read_csv('features/legitimate_test.csv')
    phish_test = pd.read_csv('features/phish_test.csv')

    test = create_scaler(pd.concat([legitimate_test, phish_test], axis=0)).sample(frac=1).reset_index(drop=True)

    X = test.drop(columns=['result_flag'])
    y = test['result_flag']

    model = load_model('models/model_C.h5')

    proba = model.predict([create_X_1(X), create_X_2(X)], batch_size=64).ravel()
    preds = (proba > 0.5).astype('int32')

    label_name = {0: 'legitimate', 1: 'phishing'}
    urls = X['url'].values

    limit = min(n_samples, len(X))
    for i in range(limit):
        print(f"{i+1}. URL: {urls[i]}")
        print(f"   True: {label_name[int(y.iloc[i])]}  "
              f"Predicted: {label_name[int(preds[i])]}  "
              f"Prob(phishing): {proba[i]:.3f}")

    print("All done.")


if __name__ == "__main__":
    main()

