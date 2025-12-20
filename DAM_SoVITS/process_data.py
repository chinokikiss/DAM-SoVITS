import re
from io import BytesIO
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from text.LangSegmenter import LangSegmenter

device = "cuda"

def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones_raw, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones_raw)
    return phones, word2ph, norm_text

def get_phones(text, language, version="v2", final=False):
    text = re.sub(r' {2,}', ' ', text)
    textlist = []
    langlist = []
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text,"ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text,"ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                langlist.append(language)
            textlist.append(tmp["text"])
    phones_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, _, norm_text = clean_text_inf(textlist[i], lang, version)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones("." + text, language, version, final=True)

    return phones

import torch
from feature_extractor import cnhubert
from SoVITS.models import SynthesizerTrn

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")
        
def load_sovits_new(sovits_path):
    f = open(sovits_path, "rb")
    meta = f.read(2)
    if meta != b"PK":
        data = b"PK" + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path, map_location="cpu", weights_only=False)

def get_sovits_weights(sovits_path):

    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"

    model_params_dict = vars(hps.model)

    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **model_params_dict,
    )

    vq_model = vq_model.to(device)

    vq_model.eval()
    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    vq_model.dec.remove_weight_norm()

    return vq_model

vq_model = get_sovits_weights("DAM_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth")
cnhubert.cnhubert_base_path = "DAM_SoVITS/pretrained_models/chinese-hubert-base"
ssl_model = cnhubert.get_model()

import os
import torch
import librosa
import argparse
import numpy as np
from funasr import AutoModel

emotion2vec = AutoModel(
    model="iic/emotion2vec_base",
    hub="ms",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_language", type=str)
    parser.add_argument("--data_save_path", type=str)
    parser.add_argument("--data_directory", type=str)
    args = parser.parse_args()

    data_language = args.data_language
    data_save_path = args.data_save_path
    data_directory = args.data_directory

    max_depth = 2
    audio_files = []
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}
    for root, dirs, files in os.walk(data_directory):
        depth = root[len(data_directory):].count(os.sep)
        if depth < max_depth:
            for filename in files:
                _, extension = os.path.splitext(filename)
                if extension.lower() in audio_extensions:
                    full_path = os.path.join(root, filename)
                    audio_files.append(os.path.abspath(full_path))
        if depth >= max_depth - 1:
            dirs.clear()
    
    print(f"总共找到{len(audio_files)}条音频")
    
    if os.path.exists(data_save_path):
        print("数据集已存在，准备添加新数据")
        data_file = np.load(data_save_path, allow_pickle=True)
        data_duration, datas = data_file
    else:
        print("数据集不存在，准备创建新数据集")
        data_duration, datas = 0, []
    for i, audio_file in enumerate(audio_files):
        lab_file = os.path.splitext(audio_file)[0] + ".lab"
        if os.path.exists(lab_file):
            with open(lab_file, mode="r", encoding="utf-8") as f:
                text = f.read().strip()

            phones = get_phones(text, data_language)
            
            wav16k, sr = librosa.load(audio_file, sr=16000)
            data_duration += librosa.get_duration(y=wav16k, sr=sr)
            wav16k = torch.from_numpy(wav16k)
            with torch.inference_mode():
                ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2).to(device)
                codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0)[0].cpu().numpy()

            emo_feature = emotion2vec.generate(audio_file, granularity="utterance", extract_embedding=True)[0]['feats']
            dirname = os.path.basename(os.path.dirname(audio_file))

            datas.append([dirname, prompt, phones, emo_feature])

            print(audio_file, f"{i+1}/{len(audio_files)}")
    
    print(f"数据集总时长: {int(data_duration/3600)}h or {int(data_duration/60)}m")
    np.save(data_save_path, [data_duration, datas])