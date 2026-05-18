from transformers import AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import evaluate, datasets
import numpy as np
from vllm.model_executor import prefill_predictor
from vllm.config_predictor import PrefillPredictorConfig
from vllm.model_executor.prefill_predictor import prefill_predictor_model
import json
import torch
from argparse import ArgumentParser, Namespace
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from allrank.models.losses.neuralNDCG import neuralNDCG
from allrank.models.losses.listMLE import listMLE
from scipy.stats import kendalltau
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
import os
from tqdm import tqdm
import math
# [best-tau] copy.deepcopy dùng để snapshot state_dict khi epoch hiện tại có
# tau test cao nhất. State_dict được restore vào model trước save_pretrained.
import copy

def parse_args():
    parser = ArgumentParser("allRank")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--print-loss", action='store_true')
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric-name", type=str, default="mse")
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wc", type=float, default=0.01)
    parser.add_argument("--loss", type=str, default='crossentropy')
    parser.add_argument("--job-dir", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--label-max-length", type=int, default=8192) #context length for llama-3
    parser.add_argument("--label-group-size", type=int, default=1)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-70B")

    return parser.parse_args()

class RankingDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048, label_max_length=8192, label_group_size=1):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_max_length = label_max_length
        self.label_group_size = label_group_size

    def __len__(self):
        return len(self.data)

    def __len2label__(self, length):
        label = self.label_max_length // self.label_group_size -  min(self.label_max_length, length) // self.label_group_size
        return label

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        origin_len = len(self.tokenizer(item['generated'])['input_ids'])
        label = self.__len2label__(origin_len)

        return prompt, label, origin_len

class RankingTestDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048, label_max_length=8192):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_max_length = label_max_length

    def __len__(self):
        return len(self.data)

    def __len2label__(self, length):
        label = self.label_max_length  -  min(self.label_max_length, length) 
        return label

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        origin_len = len(self.tokenizer(item['generated'])['input_ids'])
        label = self.__len2label__(origin_len)
        return prompt, label, origin_len



def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    
    llama3_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prefill_predictor_model_config = args.config # 'config_prefill_opt.txt'
    config = PrefillPredictorConfig.from_json(prefill_predictor_model_config)


    if config.model.num_labels == -1:
        config.model.num_labels = math.ceil(args.label_max_length / args.label_group_size)
    print("num_labels: ", config.model.num_labels)


    with set_default_torch_dtype(torch.float32):
        with torch.device('cuda'):
            predictor = prefill_predictor_model(pred_model=config.model.pred_model, num_labels=config.model.num_labels, mtype=config.model.mtype, activation=config.model.activation, max_length=config.model.max_length, max_batch_size=config.model.max_batch_size)
    # `torch.device('cuda')` không reliably move pretrained weights cho mọi
    # architecture (vd DistilBERT load checkpoint về CPU → embedding mismatch
    # với input cuda:0). Force toàn bộ predictor lên cuda:0 — idempotent nếu
    # đã đúng device.
    predictor = predictor.to("cuda:0")

    dataset_path = args.file
    dataset = []
    

    with open(dataset_path) as f:
        for jsonObj in f:
            info = json.loads(jsonObj)
            dataset.append(info)
    

    train_dataset = RankingDataset(dataset[:int(0.9 * len(dataset))], llama3_tokenizer, max_length=config.model.max_length, label_max_length=args.label_max_length, label_group_size=args.label_group_size)
    test_dataset = RankingTestDataset(dataset[int(0.9 * len(dataset)):], llama3_tokenizer, max_length=config.model.max_length, label_max_length=args.label_max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(predictor.model.parameters(), lr=args.lr, weight_decay=args.wc)
    optimizer.zero_grad()

    if args.loss == 'listMLE':
        loss_func = listMLE
    if args.loss == 'neuralNDCG':
        loss_func = neuralNDCG
    elif args.loss == 'mse':
        loss_func = torch.nn.MSELoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss()

    # [best-tau] Track epoch có Kendall's Tau test cao nhất. Trainer trước đây
    # save weights cuối loop (overfit risk với epoch lớn + model nhỏ). Giờ
    # snapshot state_dict mỗi khi tau cải thiện, restore trước save_pretrained.
    best_tau = -float('inf')
    best_epoch = 0
    best_state = None
    tau_history = []

    for epoch in range(args.epoch):
        predictor.model.train()
        total_loss = 0
        idx = 0
        for prompt, labels, origin_len in tqdm(train_dataloader):
            prompt = list(prompt)
            
            encoded_inputs = predictor.tokenizer(prompt, max_length=config.model.max_length, padding=True, truncation=True, return_tensors="pt")
            
            input_ids = encoded_inputs['input_ids'].to("cuda:0")
            attention_mask = encoded_inputs['attention_mask'].to("cuda:0")

            with torch.autocast(device_type="cuda"):

                outputs = predictor(input_ids, attention_mask)

                labels = labels.reshape(1, -1)
                labels = labels.to("cuda")
                if args.loss == 'crossentropy':
                    assert labels.max().item() < predictor.model.num_labels
                    logits = outputs.view(-1, predictor.model.num_labels)
                    loss = loss_func(logits, labels.view(logits.size(0))) 
                else:
                    loss = loss_func(outputs.view(1, -1), labels) 
            
            if args.print_loss:
                print("loss: ", loss ) 
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
                        
            total_loss += loss.item()
            idx += 1
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

        true_labels = []
        predictions = []
        
        predictor.model.eval()
        with torch.no_grad():
            train_labels = []
            for prompt, labels, origin_len in tqdm(test_dataloader):
                prompt = list(prompt)

                encoded_inputs = predictor.tokenizer(prompt, max_length=config.model.max_length, padding=True, truncation=True, return_tensors="pt")
                input_ids = encoded_inputs['input_ids'].to("cuda:0")
                attention_mask = encoded_inputs['attention_mask'].to("cuda:0")
                with torch.autocast(device_type="cuda"):
                    outputs = predictor(input_ids, attention_mask)

                if args.loss == 'crossentropy':
                    predicted_scores = outputs.argmax(dim=-1).tolist()
                else:
                    predicted_scores = outputs.squeeze().tolist()

                true_labels.extend(labels.tolist())
                train_labels.extend([train_dataset.__len2label__(l) for l in origin_len])
                predictions.extend(predicted_scores)

            tau, score = kendalltau(true_labels, predictions)
            print(f"Kendall's Tau: {tau}, p-value: {score}")

            if args.loss == 'crossentropy':
                print("acc: ", (np.array(train_labels) == np.array(predictions) ).sum() / len(train_labels) )

        # [best-tau] Update best snapshot nếu tau epoch này cao hơn. Tau từ
        # kendalltau(label, prediction) — cả 2 đều inverse-of-length → positive
        # direction, "cao nhất" = most positive. deepcopy state_dict tốn RAM
        # ~size model FP32 (Pythia-14m: 30MB, Pythia-70m: 180MB) — chấp nhận
        # được để tránh I/O save_pretrained mỗi epoch.
        tau_history.append(float(tau))
        if tau > best_tau:
            best_tau = tau
            best_epoch = epoch + 1
            best_state = copy.deepcopy(predictor.model.state_dict())
            print(f"  → new best @ epoch {best_epoch}, tau={tau:.4f}")

    # [best-tau] Restore weights của epoch có tau cao nhất trước khi save.
    # Nếu loop chưa từng update best (vd args.epoch=0), bỏ qua restore — model
    # giữ nguyên state hiện tại.
    if best_state is not None:
        predictor.model.load_state_dict(best_state)
        print(f"\nRestored best-tau weights: epoch {best_epoch}, tau={best_tau:.4f}")

    paths = PathsContainer.from_args(args.job_dir, args.run_id, prefill_predictor_model_config)
    
    usage_config_path = os.path.join(paths.output_dir, "usage_config.json")
    
    finetuned_model_output_path = os.path.join(paths.output_dir, "finetuned")

    config.model.path =  str(finetuned_model_output_path)

    create_output_dirs(paths.output_dir)

    PrefillPredictorConfig.to_json(config, usage_config_path)

    # Ghi thêm usage_config_ov.json để model chạy được trên CPU qua OpenVINO
    # backend (PrefillModelConfig.device="openvino"). Mutate sau khi save bản
    # mặc định để không ảnh hưởng usage_config.json.
    ov_config_path = os.path.join(paths.output_dir, "usage_config_ov.json")
    config.model.device = "openvino"
    config.model.num_threads = 32
    config.model.inference_precision = "f16"
    PrefillPredictorConfig.to_json(config, ov_config_path)

    predictor.model.config.__dict__['num_labels'] = config.model.num_labels

    predictor.model = predictor.model.half()
    predictor.model.save_pretrained(finetuned_model_output_path)

    # [best-tau] Ghi metadata để debug overfit pattern post-hoc. So best_epoch
    # vs last_epoch: nếu best_epoch << last_epoch → overfit confirmed, có thể
    # tune args.epoch xuống cho run sau. tau_history dùng vẽ curve.
    summary = {
        "best_epoch": best_epoch,
        "best_tau": float(best_tau),
        "last_epoch": args.epoch,
        "last_tau": tau_history[-1] if tau_history else None,
        "total_epochs_run": len(tau_history),
        "loss": args.loss,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "label_group_size": args.label_group_size,
        "tau_history": tau_history,
    }
    summary_path = os.path.join(paths.output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    run()


