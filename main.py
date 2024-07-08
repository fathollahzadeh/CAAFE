from functools import partial
from tabpfn import TabPFNClassifier
import torch
import yaml
from argparse import ArgumentParser
from util.FileHandler import read_text_file_line_by_line
from util.Config import set_config
from sklearn.ensemble import RandomForestClassifier
from caafe import CAAFEClassifier
from caafe.preprocessing import make_datasets_numeric
from util.FileHandler import reader_CSV
from caafe.data import refactor_openml_description, get_X_y
from sklearn.metrics import accuracy_score, f1_score, log_loss
from util.LogResults import LogResults
import time


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--dataset-description', type=str, default="yes")
    parser.add_argument('--prompt-number-iteration', type=int, default=1)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--delay', type=int, default=20)
    parser.add_argument('--result-output-path', type=str, default="/tmp/results.csv")
    parser.add_argument('--classifier', type=str, default="TabPFN")
    parser.add_argument('--system-log', type=str, default="/tmp/caafe-system-log.dat")

    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    with open(args.metadata_path, "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.dataset_name = config_data[0].get('name')
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            try:
                args.data_source_train_path = args.data_path + "/" + config_data[0].get('dataset').get('train').replace("{user}/", "")
                args.data_source_test_path = args.data_path + "/" + config_data[0].get('dataset').get('test').replace("{user}/","")
            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)

    if args.llm_model is None:
        raise Exception("--llm-model is a required parameter!")

    if args.prompt_number_iteration is None:
        args.prompt_number_iteration = 1

    if args.dataset_description.lower() == "yes":
        dataset_description_path = args.metadata_path.replace(".yaml", ".txt")
        args.description = read_text_file_line_by_line(fname=dataset_description_path)
        args.dataset_description = 'Yes'
    else:
        args.description = None
        args.dataset_description = 'No'

    return args


if __name__ == "__main__":

    args = parse_arguments()
    set_config(model=args.llm_model, delay=args.delay, system_log=args.system_log)

    time_start = time.time()

    description = refactor_openml_description(args.description)
    df_train = reader_CSV(args.data_source_train_path)
    df_test = reader_CSV(args.data_source_test_path)

    df_train, df_test, _ = make_datasets_numeric(df_train = df_train, df_test = df_test,
                                              target_column=args.target_attribute)

    _, train_y = get_X_y(df_train, args.target_attribute)
    _, test_y = get_X_y(df_test, args.target_attribute)

    time_end = time.time()
    extra_time = time_end - time_start

    clf_no_feat_eng = None
    if args.classifier == "TabPFN":
        clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'),
                                           N_ensemble_configurations=4)
        clf_no_feat_eng.max_num_classes = 150
        clf_no_feat_eng.max_num_features = 500
        clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

    elif args.classifier == "RandomForest":
        clf_no_feat_eng = RandomForestClassifier(max_leaf_nodes=2)

    caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                                llm_model=args.llm_model,
                                iterations=args.prompt_number_iteration)

    caafe_clf.fit_pandas(df_train,
                         target_column_name=args.target_attribute,
                         dataset_description=description)

    pred_test = caafe_clf.predict(df_test)
    pred_train = caafe_clf.predict(df_train)

    acc_test = accuracy_score(pred_test, test_y)
    acc_train = accuracy_score(pred_train, train_y)

    for i in range(1, args.prompt_number_iteration+1):
        try:
            performance = caafe_clf.performance_results[i]
            log_results = LogResults(dataset_name=args.dataset_name,
                                     task_type=args.task_type,
                                     time_total=extra_time + caafe_clf.time_execution,
                                     classifier=args.classifier,
                                     status="True",
                                     time_execution=caafe_clf.time_execution,
                                     number_iteration=i,
                                     has_description=args.dataset_description,
                                     llm_model=args.llm_model,
                                     train_auc=performance["train_auc_ovo"],
                                     train_auc_ovo=performance["train_auc_ovo"],
                                     train_auc_ovr=performance["train_auc_ovr"],
                                     train_accuracy=performance["train_acc"],
                                     test_auc=performance["test_auc_ovo"],
                                     test_auc_ovo=performance["test_auc_ovo"],
                                     test_auc_ovr=performance["test_auc_ovr"],
                                     test_accuracy=performance["test_acc"],
                                     prompt_token_count=caafe_clf.prompt_number_of_tokens,
                                     all_token_count=performance['number_of_tokens']
                                     )
            log_results.save_results(result_output_path=args.output_path)
        except Exception as e:
            print(f"iteration {i} missed!!")