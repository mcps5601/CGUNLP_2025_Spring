from torch.utils.data import Dataset
from datasets import load_dataset

class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation"]
        data_name = "SemEvalWorkshop/sem_eval_2014_task_1"
        self.data = load_dataset(data_name, split=split, cache_dir="./cache/").to_list()

        # 有些中文的標點符號在tokenizer編碼以後會變成[UNK]，所以將其換成英文標點
        self.token_replacement = [
            ["：" , ":"],
            ["，" , ","],
            ["“" , "\""],
            ["”" , "\""],
            ["？" , "?"],
            ["……" , "..."],
            ["！" , "!"]
        ]

    def __getitem__(self, index):
        d = self.data[index]
        # 把中文標點替換掉
        for k in ['premise', 'hypothesis']:
            for tok in self.token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)
    