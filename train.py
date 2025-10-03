import os
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from src.preprocess import ensure_nltk_resources, load_intents, prepare_training_data


INTENTS_PATH = os.path.join("data", "intents.json")
MODEL_PATH = "model.h5"
WORDS_PATH = "words.pkl"
CLASSES_PATH = "classes.pkl"


def main() -> None:
    ensure_nltk_resources()
    intents = load_intents(INTENTS_PATH)
    words, classes, documents, x_train, y_train = prepare_training_data(intents)

    model = Sequential()
    model.add(Dense(128, input_shape=(len(words),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation="softmax"))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=200, batch_size=8, verbose=1)

    model.save(MODEL_PATH)
    with open(WORDS_PATH, "wb") as f:
        pickle.dump(words, f)
    with open(CLASSES_PATH, "wb") as f:
        pickle.dump(classes, f)

    print(f"Training complete. Saved: {MODEL_PATH}, {WORDS_PATH}, {CLASSES_PATH}")


if __name__ == "__main__":
    main()

