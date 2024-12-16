import numpy as np
import keras
from keras import ops
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
MODEL_PATH = "./models/"

# Define the distillation loss
def distillation_loss(y_true, y_pred_student, y_pred_teacher, alpha=0.7, temperature=3):
    """
    Compute the distillation loss.
    """
    hard_loss = keras.losses.categorical_crossentropy(y_true, y_pred_student)
    
    teacher_probs = ops.softmax(y_pred_teacher / temperature)
    student_probs = ops.softmax(y_pred_student / temperature)
    soft_loss = ops.mean(keras.losses.kl_divergence(teacher_probs, student_probs))
    
    return alpha * hard_loss + (1 - alpha) * soft_loss

def create_teacher_model(input_shape, num_classes):
    teacher_input = Input(shape=input_shape)
    x = Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu")(teacher_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    teacher_output = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=teacher_input, outputs=teacher_output)

def create_student_model(input_shape, num_classes):
    student_input = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(student_input)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    student_output = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=student_input, outputs=student_output)

def train_model_with_cmkd(seed, alpha=0.7, temperature=3):
    # Load preprocessed data
    X = np.load(f'./X/X{seed}.npy')
    y = np.load(f'./y/y{seed}.npy')

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_no = 1
    teacher_accuracies = []
    student_accuracies = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

        y_train_cat = keras.utils.to_categorical(y_train, num_classes=200)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes=200)

        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        teacher_model = create_teacher_model(input_shape, 200)
        teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        teacher_callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(f'{MODEL_PATH}teacher_fold_{fold_no}.keras', save_best_only=True)
        ]

        teacher_model.fit(X_train, y_train_cat,
                        validation_data=(X_val, y_val_cat),
                        epochs=50, batch_size=32, verbose=1,
                        callbacks=teacher_callbacks)

        teacher_scores = teacher_model.evaluate(X_val, y_val_cat, verbose=0)
        teacher_accuracies.append(teacher_scores[1] * 100)

        # Load best teacher model
        teacher_model = keras.saving.load_model(f'{MODEL_PATH}teacher_fold_{fold_no}.keras')

        # Train student model
        student_model = create_student_model(input_shape, 200)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        for epoch in range(50):
            for i in range(0, len(X_train), 32):
                X_batch = X_train[i:i + 32]
                y_batch = y_train_cat[i:i + 32]

                with keras.ops.GradientTape() as tape:
                    y_pred_teacher = teacher_model(X_batch, training=False)
                    y_pred_student = student_model(X_batch, training=True)
                    loss = distillation_loss(y_batch, y_pred_student, y_pred_teacher, alpha, temperature)

                grads = tape.gradient(loss, student_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, student_model.trainable_variables))

            val_loss, val_acc = student_model.evaluate(X_val, y_val_cat, verbose=0)

        # Save student model
        keras.saving.save_model(student_model, f'{MODEL_PATH}student_fold_{fold_no}.keras')

        student_scores = student_model.evaluate(X_val, y_val_cat, verbose=0)
        student_accuracies.append(student_scores[1] * 100)

        fold_no += 1

    # Save best student model
    best_student_accuracy = max(student_accuracies)
    best_fold = student_accuracies.index(best_student_accuracy) + 1
    best_student_model = create_student_model(input_shape, 200)
    best_student_model = keras.saving.load_model(f'{MODEL_PATH}student_fold_{best_fold}.keras')
    keras.saving.save_model(best_student_model, MODEL_PATH+ 'best_model.keras')

    return best_student_model
