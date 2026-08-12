"""
================================================================================
MOVIE GENRE CLASSIFICATION - CODSOFT TASK 1
================================================================================
Intern: Vishal Kadwe | ID: BY26RY229988
Optimized for: Low-end laptops (i3, 8GB RAM, No GPU)
Run: python movie_genre_task1.py
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No GUI needed - saves memory
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# Suppress matplotlib font warnings
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score)
from sklearn.preprocessing import LabelEncoder

print("="*70)
print("  MOVIE GENRE CLASSIFICATION - CODSOFT TASK 1")
print("  Intern: Vishal Kadwe | ID: BY26RY229988")
print("="*70)
print(f"\n  Started at: {datetime.now().strftime('%H:%M:%S')}")
print("="*70)

# ==============================================================================
# STEP 1: CREATE DEMO DATASET (No external files needed!)
# ==============================================================================
print("\n[STEP 1/8] Creating demo dataset...")

genre_samples = {
    'drama': [
        "A young man struggles to find his place in the world after losing his father in a tragic accident.",
        "Two sisters reunite after years apart and confront their painful family history.",
        "An aging professor reflects on his life choices while mentoring a promising student.",
        "A war veteran returns home and must rebuild his relationship with his estranged wife.",
        "A small-town teacher inspires her students to overcome poverty and pursue their dreams.",
        "A couple faces the ultimate test when one is diagnosed with a terminal illness.",
        "An artist battles depression while trying to complete his masterpiece before a gallery opening.",
        "A mother fights against all odds to protect her children from an abusive environment.",
        "A musician loses his hearing and must find new meaning in his art and life.",
        "Three generations of women live under one roof and learn to understand each other.",
        "A father and son embark on a road trip to mend their broken relationship.",
        "A nurse discovers a dark secret at the hospital where she works.",
        "An immigrant family struggles to adapt while preserving their cultural identity.",
        "A retired boxer trains a young fighter while battling his own demons.",
        "A journalist uncovers a conspiracy that puts her family in danger."
    ],
    'comedy': [
        "A group of friends plans a disastrous bachelor party in Las Vegas with hilarious consequences.",
        "Two rival coworkers are forced to team up for a company talent show with unexpected results.",
        "A bumbling detective tries to solve a murder case while dealing with his own personal chaos.",
        "A family vacation goes completely wrong when their RV breaks down in the middle of nowhere.",
        "An awkward teenager tries to survive high school while navigating first love and friendships.",
        "A chef accidentally becomes a viral sensation after a cooking video goes hilariously wrong.",
        "Two strangers swap lives for a week and discover the humor in each other's worlds.",
        "A retired superhero tries to live a normal life but keeps getting pulled back into action.",
        "A wedding planner's own wedding turns into a comedy of errors with her eccentric family.",
        "A talking dog helps his owner find true love in the most unexpected ways.",
        "A man pretends to be a doctor to impress his date and ends up in a real emergency.",
        "A group of seniors starts a garage band and becomes an internet sensation.",
        "A mailman discovers he can talk to animals and chaos ensues in the neighborhood.",
        "A fake psychic accidentally solves real crimes and must keep up the charade.",
        "A robot butler malfunctions and turns a quiet household into a circus."
    ],
    'action': [
        "A special forces agent must stop a terrorist group from detonating a nuclear device in the city.",
        "An ex-convict is recruited for a dangerous heist that turns into a fight for survival.",
        "A skilled martial artist seeks revenge against the crime syndicate that killed his brother.",
        "A pilot must land a damaged plane while terrorists try to take control from the ground.",
        "An elite sniper is framed for assassination and must clear his name while being hunted.",
        "A race car driver gets caught in an international espionage plot during a championship.",
        "A retired spy is pulled back into service when his daughter is kidnapped by enemies.",
        "A bomb disposal expert races against time to defuse explosives planted across the city.",
        "A bodyguard must protect a witness from a ruthless assassin during a cross-country chase.",
        "An undercover cop infiltrates a criminal organization and faces the ultimate loyalty test.",
        "A hacker discovers a plot to crash the global financial system and must stop it alone.",
        "A ship captain battles pirates while protecting a cargo of vital medical supplies.",
        "A parkour expert uses his skills to evade corrupt police in a dystopian city.",
        "A firefighter leads a team into a burning skyscraper to rescue hostages.",
        "A soldier stranded behind enemy lines must guide his squad to safety."
    ],
    'horror': [
        "A family moves into a haunted house and begins experiencing terrifying supernatural events.",
        "A group of teenagers accidentally awakens an ancient evil while camping in the woods.",
        "A woman discovers that her new apartment building has a dark and murderous history.",
        "A scientist's experiment goes wrong, creating a monster that stalks the laboratory.",
        "A small town is terrorized by a creature that only appears during the full moon.",
        "A paranormal investigator faces her greatest fear while exploring an abandoned asylum.",
        "A cursed video tape causes anyone who watches it to die within seven days.",
        "A group of survivors must navigate through a zombie-infested city to reach safety.",
        "A doll comes to life and terrorizes the family that brought it into their home.",
        "A remote cabin in the woods becomes a nightmare when strangers arrive with sinister intentions.",
        "A mirror in an antique shop shows visions of a terrifying parallel world.",
        "A sleep researcher discovers a demon that feeds on human nightmares.",
        "A cruise ship drifts into mysterious fog where passengers begin to vanish.",
        "A photographer realizes the subjects in her photos are dying in real life.",
        "A subway tunnel leads to an underground labyrinth inhabited by unknown creatures."
    ],
    'romance': [
        "Two strangers meet on a train and spend a magical day together before parting ways.",
        "A writer falls in love with the subject of her latest biography against all odds.",
        "Childhood sweethearts reunite after twenty years and discover their love never faded.",
        "A royal prince falls for a commoner, creating a scandal in the kingdom.",
        "Two rival chefs compete for a prestigious award while falling for each other.",
        "A widow finds love again with the man who helped rebuild her family home.",
        "A musician and a dancer find their paths crossing repeatedly in the city of dreams.",
        "An arranged marriage turns into a beautiful love story between two unlikely partners.",
        "A time traveler falls in love with someone from a different century.",
        "Two pen pals finally meet after years of writing and discover they are perfect for each other.",
        "A florist and a baker fall in love while competing in a town festival.",
        "A pilot and an air traffic controller fall in love over radio conversations.",
        "A librarian discovers love letters hidden in old books from a secret admirer.",
        "A storm strands two enemies in a cabin where they discover unexpected feelings.",
        "A photographer falls for her mysterious subject who appears only at sunset."
    ]
}

# Build datasets
train_lines = []
test_lines = []
id_counter = 1

for genre, descriptions in genre_samples.items():
    train_descs = descriptions[:10]   # 10 for training
    test_descs = descriptions[10:15]  # 5 for testing

    for desc in train_descs:
        train_lines.append(f"{id_counter} ::: Movie {id_counter} ::: {genre} ::: {desc}")
        id_counter += 1

    for desc in test_descs:
        test_lines.append(f"{id_counter} ::: Movie {id_counter} ::: {desc}")
        id_counter += 1

# Save files
with open('train_data.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_lines))
with open('test_data.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_lines))

print(f"  Training samples: {len(train_lines)}")
print(f"  Test samples: {len(test_lines)}")
print(f"  Genres: {list(genre_samples.keys())}")

# ==============================================================================
# STEP 2: LOAD DATA
# ==============================================================================
print("\n[STEP 2/8] Loading data...")

def load_data(train_path, test_path=None):
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ::: ')
                if len(parts) >= 4:
                    train_data.append({'ID': parts[0], 'TITLE': parts[1], 
                                      'GENRE': parts[2], 'DESCRIPTION': parts[3]})
    df_train = pd.DataFrame(train_data)

    df_test = None
    if test_path:
        test_data = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ::: ')
                    if len(parts) >= 3:
                        test_data.append({'ID': parts[0], 'TITLE': parts[1], 
                                         'DESCRIPTION': parts[2]})
        df_test = pd.DataFrame(test_data)

    return df_train, df_test

df_train, df_test = load_data('train_data.txt', 'test_data.txt')
print(f"  Loaded {df_train.shape[0]} training samples")
print(f"  Loaded {df_test.shape[0]} test samples")
print(f"  Genres found: {df_train['GENRE'].unique().tolist()}")

# ==============================================================================
# STEP 3: EDA (Lightweight)
# ==============================================================================
print("\n[STEP 3/8] Generating EDA charts...")

# Genre distribution
plt.figure(figsize=(8, 5))
genre_counts = df_train['GENRE'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
bars = plt.bar(genre_counts.index, genre_counts.values, color=colors, edgecolor='black')
plt.title('Distribution of Movie Genres', fontsize=13, fontweight='bold')
plt.xlabel('Genre', fontsize=11)
plt.ylabel('Count', fontsize=11)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, 
             f'{int(height)}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('genre_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: genre_distribution.png")

# Description length
df_train['word_count'] = df_train['DESCRIPTION'].apply(lambda x: len(x.split()))

plt.figure(figsize=(8, 4))
plt.hist(df_train['word_count'], bins=15, color='coral', edgecolor='black', alpha=0.7)
plt.title('Description Word Count Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.axvline(df_train['word_count'].mean(), color='red', linestyle='--', 
           label=f"Mean: {df_train['word_count'].mean():.0f}")
plt.legend()
plt.tight_layout()
plt.savefig('description_length_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: description_length_distribution.png")

# ==============================================================================
# STEP 4: TEXT PREPROCESSING (No NLTK - uses basic Python to save memory)
# ==============================================================================
print("\n[STEP 4/8] Preprocessing text...")

STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'the', 'this', 'but', 'they', 'have', 'had', 'what', 'said', 'each', 'which',
    'she', 'do', 'how', 'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these',
    'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two',
    'more', 'very', 'after', 'words', 'its', 'just', 'where', 'most', 'know', 'get',
    'through', 'back', 'much', 'before', 'too', 'any', 'same', 'tell', 'does',
    'set', 'three', 'want', 'air', 'well', 'also', 'play', 'small', 'end', 'put',
    'home', 'read', 'hand', 'port', 'large', 'spell', 'add', 'even', 'land',
    'here', 'must', 'big', 'high', 'such', 'follow', 'act', 'why', 'ask', 'men',
    'change', 'went', 'light', 'kind', 'off', 'need', 'house', 'picture', 'try',
    'us', 'again', 'animal', 'point', 'mother', 'world', 'near', 'build', 'self',
    'earth', 'father', 'head', 'stand', 'own', 'page', 'should', 'country',
    'found', 'answer', 'school', 'grow', 'study', 'still', 'learn', 'plant',
    'cover', 'food', 'sun', 'four', 'between', 'state', 'keep', 'eye', 'never',
    'last', 'let', 'thought', 'city', 'tree', 'cross', 'farm', 'hard', 'start',
    'might', 'story', 'saw', 'far', 'sea', 'draw', 'left', 'late', 'run',
    'while', 'press', 'close', 'night', 'real', 'life', 'few', 'north', 'open',
    'seem', 'together', 'next', 'white', 'children', 'begin', 'got', 'walk',
    'example', 'ease', 'paper', 'group', 'always', 'music', 'those', 'both',
    'mark', 'often', 'letter', 'until', 'mile', 'river', 'car', 'feet', 'care',
    'second', 'book', 'carry', 'took', 'science', 'eat', 'room', 'friend',
    'began', 'idea', 'fish', 'mountain', 'stop', 'once', 'base', 'hear',
    'horse', 'cut', 'sure', 'watch', 'color', 'face', 'wood', 'main', 'enough',
    'plain', 'girl', 'usual', 'young', 'ready', 'above', 'ever', 'red', 'list',
    'though', 'feel', 'talk', 'bird', 'soon', 'body', 'dog', 'family', 'direct',
    'pose', 'leave', 'song', 'measure', 'door', 'product', 'black', 'short',
    'numeral', 'class', 'wind', 'question', 'happen', 'complete', 'ship',
    'area', 'half', 'rock', 'order', 'fire', 'south', 'problem', 'piece',
    'told', 'knew', 'pass', 'since', 'top', 'whole', 'king', 'space',
    'heard', 'best', 'hour', 'better', 'during', 'hundred', 'five',
    'remember', 'step', 'early', 'hold', 'west', 'ground', 'interest',
    'reach', 'fast', 'verb', 'sing', 'listen', 'six', 'table', 'travel',
    'less', 'morning', 'ten', 'simple', 'several', 'vowel', 'toward',
    'war', 'lay', 'against', 'pattern', 'slow', 'center', 'love',
    'person', 'money', 'serve', 'appear', 'road', 'map', 'rain',
    'rule', 'govern', 'pull', 'cold', 'notice', 'voice', 'unit',
    'power', 'town', 'fine', 'certain', 'fly', 'fall', 'lead',
    'cry', 'dark', 'machine', 'note', 'wait', 'plan', 'figure',
    'star', 'box', 'noun', 'field', 'rest', 'correct', 'able',
    'pound', 'done', 'beauty', 'drive', 'stood', 'contain',
    'front', 'teach', 'week', 'final', 'gave', 'green', 'oh',
    'quick', 'develop', 'ocean', 'warm', 'free', 'minute',
    'strong', 'special', 'mind', 'behind', 'clear', 'tail',
    'produce', 'fact', 'street', 'inch', 'multiply', 'nothing',
    'course', 'stay', 'wheel', 'full', 'force', 'blue', 'object',
    'decide', 'surface', 'deep', 'moon', 'island', 'foot', 'system',
    'busy', 'test', 'record', 'boat', 'common', 'gold', 'possible',
    'plane', 'stead', 'dry', 'wonder', 'laugh', 'thousand', 'ago',
    'ran', 'check', 'game', 'shape', 'equate', 'hot', 'miss',
    'brought', 'heat', 'snow', 'tire', 'bring', 'yes', 'distant',
    'fill', 'east', 'paint', 'language', 'among'
}

def preprocess_text(text):
    """Lightweight text preprocessing without NLTK."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words)

df_train['cleaned'] = df_train['DESCRIPTION'].apply(preprocess_text)
df_test['cleaned'] = df_test['DESCRIPTION'].apply(preprocess_text)

print(f"  Example:")
print(f"    Original: {df_train['DESCRIPTION'].iloc[0][:60]}...")
print(f"    Cleaned:  {df_train['cleaned'].iloc[0][:60]}...")

# ==============================================================================
# STEP 5: TF-IDF FEATURE EXTRACTION (Optimized for low RAM)
# ==============================================================================
print("\n[STEP 5/8] Extracting TF-IDF features...")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_train['GENRE'])

X_train_text, X_val_text, y_train, y_val = train_test_split(
    df_train['cleaned'], y, test_size=0.2, random_state=42, stratify=y
)

# Reduced features for low-end laptops
vectorizer = TfidfVectorizer(
    max_features=2000,      # Reduced from 5000 for low RAM
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)

print(f"  Training features: {X_train.shape}")
print(f"  Validation features: {X_val.shape}")
print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

# ==============================================================================
# STEP 6: TRAIN 3 MODELS
# ==============================================================================
print("\n[STEP 6/8] Training models...")

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=500, C=1.0, random_state=42),
    'SVM': LinearSVC(C=1.0, max_iter=1000, random_state=42)
}

results = {}
best_acc = 0
best_model_name = ''
best_model = None

print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weighted':>12}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12}")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')

    results[name] = {'model': model, 'accuracy': acc, 'f1_macro': f1_macro,
                     'f1_weighted': f1_weighted, 'predictions': y_pred}

    print(f"  {name:<25} {acc:>10.4f} {f1_macro:>10.4f} {f1_weighted:>12.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model = model

print(f"\n  Best Model: {best_model_name} (Accuracy: {best_acc:.4f})")

# ==============================================================================
# STEP 7: GENERATE CHARTS
# ==============================================================================
print("\n[STEP 7/8] Generating evaluation charts...")

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for idx, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_val, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=label_encoder.classes_,
               yticklabels=label_encoder.classes_,
               ax=axes[idx], cbar=False, annot_kws={'size': 10})
    axes[idx].set_title(f'{name}\nAcc: {result["accuracy"]:.3f}', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=9)
    axes[idx].set_ylabel('Actual', fontsize=9)
    axes[idx].tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: confusion_matrices.png")

# Model comparison
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(results))
width = 0.25
metrics = ['accuracy', 'f1_macro', 'f1_weighted']
metric_labels = ['Accuracy', 'F1-Macro', 'F1-Weighted']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
    values = [results[m][metric] for m in results.keys()]
    bars = ax.bar(x + i*width, values, width, label=label, color=color, 
                  edgecolor='black', alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Models', fontsize=11, fontweight='bold')
ax.set_ylabel('Score', fontsize=11, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(list(results.keys()), fontsize=10)
ax.legend(fontsize=10, loc='lower right')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: model_comparison.png")

# ==============================================================================
# STEP 8: TEST PREDICTIONS
# ==============================================================================
print("\n[STEP 8/8] Generating test predictions...")

X_test = vectorizer.transform(df_test['cleaned'])
test_preds = best_model.predict(X_test)
predicted_genres = label_encoder.inverse_transform(test_preds)

results_df = pd.DataFrame({
    'ID': df_test['ID'],
    'TITLE': df_test['TITLE'],
    'DESCRIPTION': df_test['DESCRIPTION'],
    'PREDICTED_GENRE': predicted_genres
})

results_df.to_csv('test_predictions.csv', index=False)

# Prediction distribution chart
plt.figure(figsize=(7, 4))
pred_counts = results_df['PREDICTED_GENRE'].value_counts()
colors2 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
plt.bar(pred_counts.index, pred_counts.values, color=colors2[:len(pred_counts)], 
        edgecolor='black')
plt.title('Predicted Genre Distribution (Test Set)', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Genre', fontsize=10)
plt.ylabel('Count', fontsize=10)
for i, v in enumerate(pred_counts.values):
    plt.text(i, v + 0.05, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('test_predictions_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: test_predictions_distribution.png")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*70)
print("  PROJECT COMPLETE!")
print("="*70)
print(f"\n  Finished at: {datetime.now().strftime('%H:%M:%S')}")
print("\n  Files Generated:")
print("    genre_distribution.png")
print("    description_length_distribution.png")
print("    confusion_matrices.png")
print("    model_comparison.png")
print("    test_predictions_distribution.png")
print("    test_predictions.csv")
print("    train_data.txt")
print("    test_data.txt")
print("\n  Models Trained:")
for name, res in results.items():
    marker = "★" if name == best_model_name else " "
    print(f"    [{marker}] {name:<22} Accuracy: {res['accuracy']:.4f}")

print("\n  Sample Predictions:")
print(results_df[['TITLE', 'PREDICTED_GENRE']].head(10).to_string(index=False))

print("\n" + "="*70)
print("  Intern: Vishal Kadwe | ID: BY26RY229988 | CodSoft ML Internship")
print("="*70)
