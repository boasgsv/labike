import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer

class ModelManager:
    def __init__(self, model_name="unsloth/Qwen2.5-32B-Instruct-bnb-4bit", max_seq_length=16384, load_in_4bit=True):
        self.tokenizer = None
        self.model = self.load_model(model_name, max_seq_length, load_in_4bit)

    def load_model(self, model_name, max_seq_length, load_in_4bit):
        model_u, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit
        )
        self.tokenizer = get_chat_template(tokenizer, chat_template="qwen25")
        model = FastLanguageModel.get_peft_model(
            model_u,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        return model

class PromptManager:
    def __init__(self):
        self.prompts = {
    "name": """You are a scientist trained in chemistry. 
You must extract information from scientific papers identifying relevant properties associated with each natural product discussed in the academic publication.
For each paper, you have to analyze the content (text) to identify the *Compound name*. It can be more than one compound name. 
Your output should be a list with the names. Return only the list, without any additional information.
""",
    "bioActivity": """You are a scientist trained in chemistry. 
You must extract information from scientific papers identifying relevant properties associated with each natural product discussed in the academic publication.
For each paper, you have to analyze the content (text) to identify the *Biological Activity*.  It can be more than one biological activity. 
Your output should be a list with the biological activities. Return only the list, without any additional information.
Options of biological activities:
['Anesthetic', 'Inhibition of Cathepsin V', 'Mutagenic', 'Antiangiogenic', 'Inhibition of Acetylcholinesterase', 'Inhibition of Cathepsin L', 'Inhibition of Myeloperoxodase', 'Antinociceptive', 'Antioxidant', 'Cell growth inhbition', 'Antichagasic', 'Antileishmanial', 'Genotoxic', 'Inhibition of Protease', 'Cytotoxic', 'Inhibition of phosphorylating electron transport', 'Antiallergenic', 'Inhibition of basal electron transport', 'Antitrypanosomal', 'Antibacterial', 'Insect antennae response', 'Antimalarial', 'Molluscicidal', 'Antifungal', 'Anxiolytic', 'Anti-inflamatory', 'Inhibition of Glycosidase', 'Anticancer', 'Inhibition of Cathepsin B', 'Insecticidal', 'Inhibition of ATP synthesis', 'Antiviral', 'Inhibition of uncoupled electron transport']
""",
    "collectionSpecie": """You are a scientist trained in chemistry. 
You must extract information from scientific papers identifying relevant properties associated with each natural product discussed in the academic publication.
For each paper, you have to analyze the content (text) to identify the *Collection Specie*, i.e., Species from which natural products were extracted. Provide the scientific name, binomial form. Family name can be provided. For example Tithonia diversifolia, Styrax camporum (Styracaceae), or Colletotrichum gloeosporioides (Phyllachoraceae).
Your output should be a list with the collection species. Return only the list, without any additional information.
""",
    "collectionType": """You are a scientist trained in chemistry. 
You must extract information from scientific papers identifying relevant properties associated with each natural product discussed in the academic publication.
For each paper, you have to analyze the content (text) to identify the Collection Type*, i.e., Collection type of the species. 
Your output should be a list with the collection type. Return only the list, without any additional information.
Options of collection types: 
['Biotransformation Product', 'Plant Isolated', 'Semisynthesis Product', 'Microorganism isolated', 'Plant Isolated, Microorganism isolated'].
""",
    "collectionSite": """You are a scientist trained in chemistry. 
You must extract information from scientific papers identifying relevant properties associated with each natural product discussed in the academic publication.
For each paper, you have to analyze the content (text) to identify the *collection Site *, i.e., the place of the collection. 
Your output should be a list with the place or places. Return only the list, without any additional information.
Options of places: 
['Sao Carlos/SP', 'Pocos De Caldas/MG', 'Araraquara/SP', 'Teodoro Sampaio/SP', 'Murici/AL', 'Maues/AM', 'Sao Sebastiao Do Passe/BA', 'Igarape-acu/PA', 'Itacoatiara/AM', 'Apore/GO', 'Peruibe/SP', 'Sao Paulo/SP', 'Pocos De Caldas/MG, Lonchocarpus atropurpureus', 'Ibirama/SC', 'Rio Claro/SP', 'Iguape/SP', 'Goiania/GO', 'Sao Miguel Arcanjo/SP', 'Rifaina/SP', 'Piracicaba/SP', 'N/A/CE', 'Urucuca/BA', 'Campinas/SP', 'Corumba/MS', 'Lavras/MG', 'Ribeirao Preto/SP, Nigrospora sphaerica', 'N/A/SP, Cedrela fissilis', 'Recife/PE', 'Cordeiropolis/SP', 'Santarem/PA', 'Itaituba/PA', 'Londrina/PR', 'Cuiaba/MT', 'Ribeirao Preto/SP', 'Itirapina/SP', 'Manaus/AM', 'Mogi Guacu/SP', 'N/A/MS', 'N/A/MG', 'Belem/PA', 'N/A/ES', 'Ibate/SP', 'N/A/PE', 'Cunha/SP', 'Rio De Janeiro/RJ', 'Chapada Dos Guimaraes/MT', 'N/A/SP', 'N/A/AM', 'Vicosa/MG', 'Rio Verde/GO', 'Pirenopolis/GO']
"""
}

    def get_prompt(self, task):
        return self.prompts[task]

class DataManager:
    def __init__(self, task, stage, fold):
        self.task = task
        self.stage = stage
        self.fold = fold
        self.df = self.load_base_df()

    def load_base_df(self):
        df = pd.read_pickle('df.pkl').reset_index(drop=True)
        return df

    def prepare_dataset(self, tokenizer, system_prompt):
        file_path = f'splits/train_doi_{self.task}_{self.fold}_{self.stage}.csv'
        doi_list = pd.read_csv(file_path)['node'].tolist()
        filtered_df = self.df[self.df['doi'].isin(doi_list)]

        def format_example(row):
            response = str(row[self.task])
            return {
                'conversations': [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": " ".join(row['texto'].split(' ')[:3000])},
                    {"from": "gpt", "value": response}
                ]
            }

        examples = [format_example(row) for _, row in filtered_df.iterrows()]
        dataset = Dataset.from_list(examples)
        dataset = standardize_sharegpt(dataset)

        def formatting_prompts_func(examples):
            texts = [
                tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                for convo in examples['conversations']
            ]
            return {"text": texts}

        return dataset.map(formatting_prompts_func, batched=True)

class TrainerPipeline:
    def __init__(self, model, tokenizer, dataset, max_seq_length):
        self.trainer = self.build_trainer(model, tokenizer, dataset, max_seq_length)

    def build_trainer(self, model, tokenizer, dataset, max_seq_length):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=25,
                learning_rate=5e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",
            ),
        )
        return train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

    def train(self):
        return self.trainer.train()

class Evaluator:
    def __init__(self, model, tokenizer, df, task, stage, fold, prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.df = df
        self.task = task
        self.stage = stage
        self.fold = fold
        self.prompt = prompt

    def inference(self, text):
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": text},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        outputs = self.model.generate(inputs, max_new_tokens=256, use_cache=True, temperature=0.00001, min_p=0.1)
        decoded = self.tokenizer.batch_decode(outputs)[0]
        parsed = decoded[decoded.find("<|im_start|>assistant"):].replace("<|im_start|>assistant", "").replace("<|im_end|>", "")
        try:
            l = eval(parsed)
        except:
            l = ['']
            print('Saida:' + parsed)
        return l

    def hits_at(self, k, true, preds):
        return np.mean([1 if t in preds[:k] else 0 for t in true])

    def evaluate(self):
        for estagio in ['1st', '2nd', '3rd', '4th']:
            file_path = f'splits/test_doi_{self.task}_{self.fold}_{estagio}.csv'
            doi_list = pd.read_csv(file_path)['node'].tolist()
            filtered_df = self.df[self.df['doi'].isin(doi_list)]
            k_dict = {"name": 50, "bioActivity": 5, "collectionSpecie": 50, "collectionType": 1, "collectionSite": 20}
            scores = []
            preds_list = []
            for _, row in filtered_df.iterrows():
                preds = self.inference(row['texto'])
                scores.append(self.hits_at(k_dict[self.task], row[self.task], preds))
                preds_list.append(preds)
            result = f"Tarefa: {self.task} | Estagio: {estagio} | Fold: {self.fold} | Hits@{k_dict[self.task]}: {np.mean(scores)}\n"
            
            with open(f"outputs_bike/{self.task}_{estagio}_{self.fold}", 'a', encoding='utf-8') as f:
                f.write(str(preds_list))
            
            with open('resultados.txt', 'a', encoding='utf-8') as f:
                f.write(result)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tarefa", required=True)
    parser.add_argument("--estagio", required=True)
    parser.add_argument("--fold", required=True)
    args = parser.parse_args()

    model_manager = ModelManager()
    prompt_manager = PromptManager()
    data_manager = DataManager(args.tarefa, args.estagio, args.fold)

    dataset = data_manager.prepare_dataset(model_manager.tokenizer, prompt_manager.get_prompt(args.tarefa))
    trainer_pipeline = TrainerPipeline(model_manager.model, model_manager.tokenizer, dataset, max_seq_length=16384)
    trainer_pipeline.train()

    evaluator = Evaluator(model_manager.model, model_manager.tokenizer, data_manager.df,
                          args.tarefa, args.estagio, args.fold, prompt_manager.get_prompt(args.tarefa))
    evaluator.evaluate()
