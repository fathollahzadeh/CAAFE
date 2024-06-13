"""
Runs the CAAFE algorithm on a dataset and saves the generated code and prompt to a file.
"""

import argparse
from functools import partial

from tabpfn.scripts import tabular_metrics
from tabpfn import TabPFNClassifier
import tabpfn
from tabpfn.scripts.tabular_baselines import clf_dict
import os
import openai
import torch
import yaml
from argparse import ArgumentParser
from util.FileHandler import save_text_file, read_text_file_line_by_line

from caafe.data import get_data_split, load_all_data
from caafe.caafe import generate_features
from caafe import  evaluate


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--data-profile-path', type=str, default=None)
    parser.add_argument('--dataset-description', type=str, default="yes")
    parser.add_argument('--prompt-representation-type', type=str, default=None)
    parser.add_argument('--prompt-samples-type', type=str, default=None)
    parser.add_argument('--prompt-number-samples', type=int, default=None)
    parser.add_argument('--prompt-number-iteration', type=int, default=1)
    parser.add_argument('--prompt-number-iteration-error', type=int, default=1)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--enable-reduction', type=bool, default=False)
    parser.add_argument('--delay', type=int, default=20)
    parser.add_argument('--result-output-path', type=str, default="/tmp/results.csv")
    parser.add_argument('--error-output-path', type=str, default="/tmp/catdb_error.csv")
    parser.add_argument('--run-code', type=bool, default=False)
    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    if args.data_profile_path is None:
        raise Exception("--data-profile-path is a required parameter!")

    # read .yaml file and extract values:
    with open(args.metadata_path, "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.dataset_name = config_data[0].get('name')
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            try:
                args.data_source_train_path = "../../../" + config_data[0].get('dataset').get('train').replace(
                    "{user}/", "")
                args.data_source_test_path = "../../../" + config_data[0].get('dataset').get('test').replace("{user}/",
                                                                                                             "")
            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)

    if args.prompt_number_samples is None:
        raise Exception("--prompt-number-samples is a required parameter!")

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

def generate_and_save_feats(i, model, seed=0, iterative_method=None, iterations=10):
    if iterative_method is None:
        iterative_method = tabpfn

    ds = cc_test_datasets_multiclass[i]

    ds, df_train, df_test, df_train_old, df_test_old = get_data_split(ds, seed)
    code, prompt, messages = generate_features(
        ds,
        df_train,
        just_print_prompt=False,
        model=model,
        iterative=iterations,
        metric_used=metric_used,
        iterative_method=iterative_method,
        display_method="print",
    )
    #
    # data_dir = os.environ.get("DATA_DIR", "data/")
    # f = open(
    #     f"{data_dir}/generated_code/{ds[0]}_{prompt_id}_{seed}_prompt.txt",
    #     "w",
    # )
    # f.write(prompt)
    # f.close()
    #
    # f = open(f"{data_dir}/generated_code/{ds[0]}_{prompt_id}_{seed}_code.txt", "w")
    # f.write(code)
    # f.close()


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=16)
    classifier.fit = partial(classifier.fit, overwrite_warning=True)
    tabpfn = partial(clf_dict["transformer"], classifier=classifier)
    metric_used = tabular_metrics.auc_metric

    ds = cc_test_datasets_multiclass[i]
    evaluate.evaluate_dataset_with_and_without_cafe(
        ds, seed, methods, metric_used, prompt_id=prompt_id
    )

    # for i in range(0, len(cc_test_datasets_multiclass)):
    #     generate_and_save_feats(i, seed=seed, iterations=iterations)
    #

    # # ==========================
    # # Parse args
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=0,
    # )
    # parser.add_argument(
    #     "--dataset_id",
    #     type=int,
    #     default=-1,
    # )
    # parser.add_argument(
    #     "--prompt_id",
    #     type=str,
    #     default="v3",
    # )
    # parser.add_argument(
    #     "--iterations",
    #     type=int,
    #     default=10,
    # )
    # args = parser.parse_args()
    # prompt_id = args.prompt_id
    # dataset_id = args.dataset_id
    # iterations = args.iterations
    # seed = args.seed
    #
    # model = "gpt-3.5-turbo" if prompt_id == "v3" else "gpt-4"
    #
    # openai.api_key = os.environ["OPENAI_API_KEY"]
    #
    # cc_test_datasets_multiclass = load_all_data()
    # if dataset_id != -1:
    #     cc_test_datasets_multiclass = [cc_test_datasets_multiclass[dataset_id]]
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # classifier = TabPFNClassifier(device=device, N_ensemble_configurations=16)
    # classifier.fit = partial(classifier.fit, overwrite_warning=True)
    # tabpfn = partial(clf_dict["transformer"], classifier=classifier)
    # metric_used = tabular_metrics.auc_metric
    #
    # for i in range(0, len(cc_test_datasets_multiclass)):
    #     generate_and_save_feats(i, seed=seed, iterations=iterations)
