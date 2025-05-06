
# load libs
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Bidirectional, Concatenate, GlobalAveragePooling1D, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_click_attribution_table(df, channel_col='channel'):
    print("Doing First and Last Click")
    # First Click
    first = (
        df
        .sort_values(by=['user_id', 'timestamp'])
        .groupby('user_id')
        .first()
        .reset_index()
        .groupby(channel_col).size().reset_index(name='conversions')
    ).sort_values('channel')
    
    first['model'] = 'First Click'

    # Last Click
    last = (
        df
        .sort_values(by=['user_id', 'timestamp'])
        .groupby('user_id')
        .last()
        .reset_index()
        .groupby(channel_col).size().reset_index(name='conversions')
    ).sort_values('channel')
    
    last['model'] = 'Last Click'
    

    print("Doing Markov Model")
    # markov model
    markov = markov_chain(df).markov_model()[0].sort_values('channel')
    markov['model'] = 'Markov Model'
    
    print("Doing RF Model")
    # RF model
    # Step 1: Initialize and prepare
    rf_model = RandomForestConversionModel()
    X_rf, y_rf = rf_model.prepare_data(df[['channel', 'device_type', 'touchpoint_number', 'converted', 'time_to_conversion_days']])

    model, pred = rf_model.train(X_rf, y_rf)

    df['pred'] = pred  # make sure prob is in the same order as df

    # Step 2: Normalize per user_id
    user_sums = df.groupby('user_id')['pred'].transform('sum')
    df['pred'] = df['pred'] / user_sums
    df['pred'] = df['pred'].fillna(0)  # in case sum was 0

    # Step 3: Group by channel and sum normalized predictions
    rf = (
        df.groupby('channel')['pred']
        .sum()
        .reset_index()
        .rename(columns={'pred': 'conversions'})
        .sort_values('channel')
        .reset_index(drop=True)
    ).sort_values('channel')
    rf['model'] = 'Random Forest'


    print("Doing LSTM Model")
    # LSTM model
    # Create and prepare
    lstm_model = LSTMConversionModel(context_window=6)
    X, y,_ = lstm_model.prepare_data(df)

    # Train
    model, pred = lstm_model.train(X, y, epochs=3)

    # Step 1: Add LSTM predictions to DataFrame
    df['pred'] = pred  # make sure prob is in the same order as df

    # Step 2: Normalize per user_id
    user_sums = df.groupby('user_id')['pred'].transform('sum')
    df['pred'] = df['pred'] / user_sums
    df['pred'] = df['pred'].fillna(0)  # in case sum was 0

    # Step 3: Group by channel and sum normalized predictions
    lstm = (
        df.groupby('channel')['pred']
        .sum()
        .reset_index()
        .rename(columns={'pred': 'conversions'})
        .sort_values('channel')
        .reset_index(drop=True)
    ).sort_values('channel')
    lstm['model'] = 'LSTM'
    
    # Combine
    combined = pd.concat([first, last, markov, lstm, rf], ignore_index=True)

    # Compute percentages within each model
    combined['percentage'] = combined.groupby('model')['conversions'].transform(lambda x: round(100 * x / x.sum(), 2))
    return combined[[channel_col, 'model', 'conversions', 'percentage']]


class markov_chain:
    def __init__(self, df):
        self.X = df
        self.transition_matrix = None
        self.state_list = None  # ordered list of states

    def build_markov_transition_matrix(self, channel_col='channel', user_col='user_id', time_col='timestamp'):
        """
        Builds a Markov transition matrix as a DataFrame.
        """
        df_sorted = self.X.sort_values(by=[user_col, time_col])
        journeys = df_sorted.groupby(user_col)[channel_col].apply(lambda x: ['start'] + list(x) + ['conversion'])

        transitions = defaultdict(lambda: defaultdict(int))
        for path in journeys:
            for i in range(len(path) - 1):
                src, tgt = path[i], path[i + 1]
                transitions[src][tgt] += 1

        all_states = sorted(set(transitions.keys()) | {tgt for dests in transitions.values() for tgt in dests})
        self.state_list = all_states

        matrix = pd.DataFrame(0.0, index=all_states, columns=all_states)

        for src, dests in transitions.items():
            total = sum(dests.values())
            if total > 0:
                for tgt, count in dests.items():
                    matrix.loc[src, tgt] = round(count / total, 6)

        self.transition_matrix = matrix
        return matrix

    def simulate_conversion_probability(self, steps=15, start_state='start', matrix_df=None):
        """
        Uses matrix multiplication to simulate conversion probability over N steps.
        """
        if matrix_df is None:
            matrix_df = self.transition_matrix
        if matrix_df is None:
            raise ValueError("Transition matrix not provided or built.")
        if start_state not in matrix_df.index:
            return 0.0

        matrix = matrix_df.values
        states = self.state_list
        start_idx = states.index(start_state)
        conv_idx = states.index('conversion')

        # Initial state vector: 100% at start, 0 elsewhere
        state_vec = np.zeros(len(states))
        state_vec[start_idx] = 1.0

        total_conv_prob = 0.0

        for _ in range(steps):
            state_vec = state_vec @ matrix
            total_conv_prob += state_vec[conv_idx]
            state_vec[conv_idx] = 0  # absorb conversion

        return round(total_conv_prob, 6)

    def markov_model(self, steps=15, transition_matrix=None):
        """
        Computes attribution via removal effect using matrix-based simulation.
        """
        if transition_matrix is not None:
            matrix = transition_matrix
        else:
            matrix = self.build_markov_transition_matrix()

        base_prob = self.simulate_conversion_probability(steps=steps, matrix_df=matrix)

        channels = [ch for ch in matrix.columns if ch not in ['start', 'conversion']]
        contributions = {}

        for ch in channels:
            mod_matrix = matrix.drop(index=ch, columns=ch, errors='ignore')
            mod_states = mod_matrix.index.tolist()
            # simulate with a filtered state list
            temp_mc = markov_chain(self.X)
            temp_mc.transition_matrix = mod_matrix
            temp_mc.state_list = mod_states
            mod_prob = temp_mc.simulate_conversion_probability(steps=steps, matrix_df=mod_matrix)
            contributions[ch] = max(0.0, base_prob - mod_prob)

        total = sum(contributions.values())
        if total == 0:
            attribution_df = pd.DataFrame([{'channel': ch, 'attribution': 0.0} for ch in contributions])
        else:
            attribution_df = pd.DataFrame([
                {'channel': ch, 'conversions': round(100 * val / total, 2)}
                for ch, val in contributions.items()
            ]).sort_values(by='conversions', ascending=False)

        return attribution_df, matrix
    
    def predict_from_path(self, path, steps=10, transition_matrix=None):
        """
        Predict conversion probability starting from the last state in a given journey path.
        
        Args:
            path (list): Journey list, e.g., ['Email', 'Social']
            steps (int): Number of steps to simulate
            transition_matrix (pd.DataFrame): Optional external transition matrix
        
        Returns:
            float: Simulated probability of conversion
        """
        if not path:
            return 0.0

        last_state = path[-1]

        matrix = transition_matrix if transition_matrix is not None else self.transition_matrix
        states = matrix.index.tolist() if matrix is not None else []

        if matrix is None or last_state not in states:
            return 0.0

        # Update state list if using external matrix
        if transition_matrix is not None:
            self.state_list = states

        return self.simulate_conversion_probability(
            start_state=last_state,
            steps=steps,
            matrix_df=matrix
        )
    
    def plot_transition_sankey(self, df):
        """
        Plots a Markov-style Sankey diagram from sequences in the input DataFrame.
        """

        journeys = (
            df.sort_values(by=['user_id', 'timestamp'])
              .groupby('user_id')['channel']
              .apply(lambda x: ['start'] + list(x) + ['conversion'])
        )

        transitions = defaultdict(lambda: defaultdict(int))
        for path in journeys:
            for i in range(len(path) - 1):
                src, tgt = path[i], path[i + 1]
                transitions[src][tgt] += 1

        all_states = sorted(set(transitions.keys()) | {tgt for dests in transitions.values() for tgt in dests})
        matrix = pd.DataFrame(0.0, index=all_states, columns=all_states)

        for src, dests in transitions.items():
            total = sum(dests.values())
            for tgt, count in dests.items():
                matrix.loc[src, tgt] = round(count / total, 4)

        # Remove 'start' if needed
        matrix_df = matrix.drop(index='start', errors='ignore')

        # Prepare Sankey data
        transitions = matrix_df.stack().reset_index()
        transitions.columns = ['source', 'target', 'value']
        transitions = transitions[transitions['value'] > 0]

        labels = pd.Index(transitions['source'].tolist() + transitions['target'].tolist()).unique().tolist()
        label_map = {label: i for i, label in enumerate(labels)}

        source = transitions['source'].map(label_map).tolist()
        target = transitions['target'].map(label_map).tolist()
        value = transitions['value'].tolist()

        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        ))

        fig.update_layout(title_text="Markov-style Channel Transitions (Sankey)", font_size=12)
        fig.show()


class LSTMConversionModel:
    def __init__(self, context_window=6):
        self.context_window = context_window
        self.model = None
        self.le = None
        self.vocab_size = 0

    def prepare_data(self, df, user_col='user_id', channel_col='channel',
                     label_col='converted', time_col='timestamp'):
        """
        Prepares padded LSTM sequences from user journeys.
        """
        df = df.sort_values(by=[user_col, time_col])
        
        self.le = LabelEncoder()
        df['channel_id'] = self.le.fit_transform(df[channel_col]) + 1  # reserve 0 for padding

        sequences = []
        labels = []

        for _, user_df in df.groupby(user_col):
            chan_ids = user_df['channel_id'].tolist()
            conv_flags = user_df[label_col].tolist()

            for i in range(len(chan_ids)):
                seq = chan_ids[:i+1]
                label = conv_flags[i]

                if len(seq) < self.context_window:
                    seq = pad_sequences([seq], maxlen=self.context_window)[0]
                else:
                    seq = seq[-self.context_window:]

                sequences.append(seq)
                labels.append(label)

        X = np.array(sequences)
        y = np.array(labels)
        self.vocab_size = df['channel_id'].max()

        return X, y, self.le

    def train(self, X, y, epochs=3, batch_size=32):
        """
        Trains the BiLSTM model with Attention using softmax output and class weighting.
        """
        # 1. Compute class weights safely
        cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        cw_dict = {int(k): float(v) for k, v in zip(np.unique(y), cw)}  # ensure native Python types

        # 2. Define model
        input_layer = Input(shape=(self.context_window,))
        embedding = Embedding(input_dim=self.vocab_size + 1, output_dim=64)(input_layer)

        lstm_1 = Bidirectional(LSTM(128, return_sequences=True))(embedding)
        drop_1 = Dropout(0.3)(lstm_1)
        lstm_2 = Bidirectional(LSTM(32, return_sequences=True))(drop_1)
        drop_2 = Dropout(0.2)(lstm_2)

        attention = Attention()([drop_2, drop_2])
        context = Dense(64, activation='relu')(attention)
        pooled = GlobalAveragePooling1D()(context)

        output = Dense(1, activation='sigmoid')(pooled)

        model = Model(inputs=input_layer, outputs=output)

        # 3. Compile with sparse categorical crossentropy (uses integer labels)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=1e-3),
                      metrics=['accuracy'])

        # 4. Train model with class weights
        model.fit(X, y,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.2,
                  shuffle=True,
                  verbose=1,
                  class_weight=cw_dict)

        # 5. Predict class 1 probability (softmax gives [prob_0, prob_1])
        y_pred_probs =  model.predict(X).flatten()

        self.model = model
        return self.model, y_pred_probs.tolist()

    def predict_path(self, model , le, path):
        """
        Predict conversion probability from a list of channel names.
        """
        for p in path:   
            encoded = le.transform(p) + 1
            if len(encoded) < self.context_window:
                padded = pad_sequences([encoded], maxlen=self.context_window)
            else:
                padded = np.array([encoded[-self.context_window:]])
        pred = model.predict(padded, verbose=0)[0][0]
        return pred
    

class RandomForestConversionModel:
    def __init__(self, features=None):
        self.features = features if features else ['channel', 'device_type', 'touchpoint_number']
        self.X_columns = None  # to store one-hot feature names for alignment

    def prepare_data(self, df, label_col='converted', test_data=False):
        """
        Prepares the encoded input features and labels from touchpoint-level data.
        """
        df = df.copy()
        df = df.dropna(subset=self.features + ([] if test_data else [label_col]))

        df_encoded = pd.get_dummies(df[self.features])

        if self.X_columns is None and not test_data:
            self.X_columns = df_encoded.columns.tolist()

        if self.X_columns:
            df_encoded = df_encoded.reindex(columns=self.X_columns, fill_value=0)

        if test_data:
            return df_encoded

        y = df[label_col].astype(int)
        return df_encoded, y

    def train(self, X, y, test_size=0.2, n_estimators=200):
        """
        Trains a Random Forest model and returns it with evaluation metrics.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f" Accuracy: {acc:.4f}")
        print(f" ROC AUC:  {auc:.4f}")

        return model, model.predict_proba(X)[:, 1]

    def predict_probabilities(self, df, model, test_data = True):
        """
        Predicts conversion probabilities using a passed model.
        """

        X = self.prepare_data(df, test_data=test_data)
        return model.predict_proba(X)[:, 1]

    def get_feature_importance(self, model):
        """
        Returns feature importances using the passed model.
        """
        if model is None or self.X_columns is None:
            raise ValueError("Model or column info missing.")

        return pd.DataFrame({
            'feature': self.X_columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)

