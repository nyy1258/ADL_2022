import json
from argparse import ArgumentParser, Namespace
import csv

def main(args):
    if(args.do_train):
        S = ["train", "valid"]
        dataset = {}
        context_file = open(args.context_dir, encoding='utf-8')
        train_file = open(args.train_dir, encoding='utf-8')
        valid_file = open(args.valid_dir, encoding='utf-8')

        context = json.load(context_file)
        
        dataset["train"] = json.load(train_file)
        dataset["valid"] = json.load(valid_file)

        for split in ["train", "valid"]:
            file = open(args.output_dir + "/" + f"pre_{split}.json", "w")
            
            for data in dataset[split]:
                D = {}
                ## swag dataset
                D["video-id"] = data["id"]
                D["fold-ind"] = data["id"]
                D["startphrase"] = data["question"]
                D["sent1"] = data["question"]
                D["sent2"] = data["question"]
                D['gold-source'] = 'gold'
                for i in range(4):
                    D[f"ending{i}"] = context[data["paragraphs"][i]]
                D["label"] = data["paragraphs"].index(data["relevant"])

                ## squad dataset
                D["answers"] = {"answer_start": [data["answer"]["start"]], "text": [data["answer"]["text"]]}
                D["context"] = context[data["relevant"]]
                D["id"] = data["id"]
                D["question"] = data["question"]    

                json.dump(D, file, ensure_ascii=False)  

            file.close()

    if(args.do_predict): 
        context_file = open(args.context_dir, encoding='utf-8')
        
        test_file = open(args.test_dir, encoding='utf-8')
        context = json.load(context_file)
        dataset = json.load(test_file)

        file = open(args.output_dir + "/" + "pre_test.json", "w")

        for data in dataset:
            D = {}
            ## swag dataset
            D["video-id"] = data["id"]
            D["fold-ind"] = data["id"]
            D["startphrase"] = data["question"]
            D["sent1"] = data["question"]
            D["sent2"] = data["question"]
            D['gold-source'] = 'gold'
            for i in range(4):
                D[f"ending{i}"] = context[data["paragraphs"][i]]
            D["label"] = 0

            ## squad dataset
            D["answers"] = {"answer_start": [0], "text": [""]}
            D["context"] = ""
            D["id"] = data["id"]
            D["question"] = data["question"]

            json.dump(D, file, ensure_ascii=False)
        file.close()
        

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--context_dir", type = str, default = "./data/context.json")
    parser.add_argument("--train_dir", type = str, default = "./data/train.json")
    parser.add_argument("--valid_dir", type = str, default = "./data/valid.json")
    parser.add_argument("--test_dir", type = str, default = "./data/test.json")
    parser.add_argument("--output_dir", type = str, default="./cache/")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)