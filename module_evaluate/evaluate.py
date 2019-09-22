import scikitplot as skplt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from module_evaluate.reference_model import *
from module_dataset.preprocess_data.hanlde_dataloader import *


def predict_test(path_data_test, path_save_model, path_model_checkpoint, type_model='cnn_classify'):
    list_predict = get_predict_dl(path_data_test, path_save_model, path_model_checkpoint, type_model)
    test_preds = []
    test_weight_preds = []
    for preds in list_predict:
        preds = torch.sigmoid(preds).cpu()

        w_pred = preds.data.numpy()  # weight_predict
        y_pred = torch.max(preds, 1)[1].numpy()  # return label

        test_preds.append(y_pred)
        test_weight_preds.append(w_pred)

    test_preds = [item for sublist in test_preds for item in sublist]
    test_weight_preds = [item for sublist in test_weight_preds for item in sublist]

    return test_preds, test_weight_preds


def error_predict_dl(path_data_test, path_save_model, path_model_checkpoint):
    x_test, y_test = load_text_label(path_data_test, is_train=True)
    y_test = list(map(int, y_test))

    test_preds, test_weight_preds = predict_test(path_data_test, path_save_model, path_model_checkpoint, type_model='cnn_classify')

    indices = [i for i in range(len(test_preds)) if test_preds[i] != y_test[i]]
    error_text = []
    error_weight = []
    error_label = []
    true_label = []
    for i in indices:
        error_text.append(x_test[i])
        error_weight.append(test_weight_preds[i])
        error_label.append(test_preds[i])
        true_label.append(y_test[i])

    report = list(zip(error_text, error_weight, error_label, true_label))
    cv_df = pd.DataFrame(report, columns=['text_data', 'predict_weight', 'predict_label', 'true_label'])
    name_error_result = os.path.join(path_save_model, "error_result.csv")
    cv_df.to_csv(name_error_result, index=False)
    return cv_df


def error_predict_ml(path_data_test, path_data_test_ft, path_save_model_ml):
    x_test, y_test = load_text_label(path_data_test, is_train=True)

    y_preds = get_predict_ml(path_data_test_ft, path_save_model_ml)

    indices = [i for i in range(len(y_test)) if y_preds[i] != y_test[i]]
    error_text = []
    error_label = []
    for i in indices:
        error_text.append(x_test[i])
        error_label.append(y_preds[i])

    report = list(zip(error_text, error_label))
    cv_df = pd.DataFrame(report, columns=['text_data', 'predict_label'])
    name_error_result = os.path.join("../module_train/save_model/model_ml/error_result.csv")
    cv_df.to_csv(name_error_result, index=False)

    return


def plot_confusion_matrix_dl(path_data_test, path_save_model, path_model_checkpoint, type_model='cnn_classify'):
    _, y_test = load_text_label(path_data_test, is_train=True)
    y_test = list(map(int, y_test))

    y_pred, _ = predict_test(path_data_test, path_save_model, path_model_checkpoint, type_model)

    error_predict_dl(path_data_test, path_save_model, path_model_checkpoint)

    labels = ['clean', 'offensive', 'hate']
    print(classification_report(y_test, y_pred, target_names=labels))

    skplt.metrics.plot_confusion_matrix(
        y_test,
        y_pred,
        figsize=(6, 6))
    plt.show()


def plot_confusion_matrix_ml(path_data_test, path_data_test_ft, path_save_model_ml):
    _, y_test = load_data_ml(path_data_test_ft, is_train=True)

    y_pred = get_predict_ml(path_data_test_ft, path_save_model_ml)

    error_predict_ml(path_data_test, path_data_test_ft, path_save_model_ml)

    labels = ['clean', 'offensive', 'hate']
    print(classification_report(y_test, y_pred, target_names=labels))

    skplt.metrics.plot_confusion_matrix(
        y_test,
        y_pred,
        figsize=(6, 6))
    plt.show()

if __name__ == '__main__':
    # test dl
    path_save_model = "../module_train/save_model/cnn_classify"
    path_data_test = "../module_dataset/dataset/data_for_train/dl/validation_dl.csv"
    path_model_checkpoint = "../module_train/save_model/cnn_classify/model_cnn_epoch_14_train_acc_0.9578_loss_0.1216_test_acc_0.9405_loss_0.295"

    label, weight = predict_test(path_data_test, path_save_model, path_model_checkpoint, type_model='cnn_classify')

    plot_confusion_matrix_dl(path_data_test, path_save_model, path_model_checkpoint, type_model='cnn_classify')
    # # test ml
    # path_data_test = "../module_dataset/dataset/data_for_train/exp_train.csv"
    # path_data_test_ft = "../module_train/save_model/model_ml/data_ft.pkl"
    # path_save_model_ml = "../module_train/save_model/model_ml/random_forest_model.pkl"
    # plot_confusion_matrix_ml(path_data_test, path_data_test_ft, path_save_model_ml)

