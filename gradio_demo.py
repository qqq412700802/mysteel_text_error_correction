# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install gradio
"""

import gradio as gr
import operator
import torch
from transformers import BertTokenizerFast, BertForMaskedLM
import os

model_path = os.path.join(os.getcwd(), 'data/macbert4csc')
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)


def ai_text(text):
    with torch.no_grad():
        outputs = model(**tokenizer([text], padding=True, return_tensors='pt'))

    def to_highlight(corrected_sent, errs):
        output = [{"entity": "纠错", "word": err[1], "start": err[2], "end": err[3]} for i, err in
                  enumerate(errs)]
        return {"text": corrected_sent, "entities": output}

    def get_errors(corrected_text, origin_text):
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
                # add unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if i >= len(corrected_text):
                continue
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    _text = tokenizer.decode(torch.argmax(outputs.logits[0], dim=-1), skip_special_tokens=True).replace(' ', '')
    corrected_text = _text[:len(text)]
    corrected_text, details = get_errors(corrected_text, text)
    print(text, ' => ', corrected_text, details)
    return to_highlight(corrected_text, details), details


if __name__ == '__main__':
    examples = [
        ['交通运输部28日发布数据，1至11月，我国完成交通固定资产投资3.5万亿元，同比增长5.8%'],
        ['具体来看，其中完成公路投资2.6万亿元，同比增长9.1%；完成水运投资1478亿元，同比增长10.2%'],
        ['1至11月，全国港口完成货物吞吐量143.1亿吨，同比增长0.7%；完成集装箱吞吐量2.7亿标箱，同比增长4.2%'],
    ]

    gr.Interface(
        ai_text,
        inputs="textbox",
        outputs=[
            gr.outputs.HighlightedText(
                label="Output",
                show_legend=True,
            ),
            gr.outputs.JSON(
                label="JSON Output"
            )
        ],
        title="钢联智能文本纠错系统",
        description="请输入待纠错的纹章",
        examples=examples
    ).launch(server_name='0.0.0.0')
