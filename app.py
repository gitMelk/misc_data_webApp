from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device



download = False
save_model_locally= False
if download:
    tokenizer = AutoTokenizer.from_pretrained("MilaNLProc/feel-it-italian-sentiment", cache_dir="data/")
    model = AutoModelForSequenceClassification.from_pretrained("MilaNLProc/feel-it-italian-sentiment", cache_dir="data/")
    model.eval()
    tokenizer_emo = AutoTokenizer.from_pretrained("MilaNLProc/feel-it-italian-emotion", cache_dir="data/")
    model_emo = AutoModelForSequenceClassification.from_pretrained("MilaNLProc/feel-it-italian-emotion", cache_dir="data/")
    model_emo.eval()
    if save_model_locally:
        model.save_pretrained('./local_models/sentiment_ITA')
        tokenizer.save_pretrained('./local_models/sentiment_ITA')
        model_emo.save_pretrained('./local_models/emotion_ITA')
        tokenizer_emo.save_pretrained('./local_models/emotion_ITA')
else:
    tokenizer = AutoTokenizer.from_pretrained("./local_models/sentiment_ITA/")
    model = AutoModelForSequenceClassification.from_pretrained("./local_models/sentiment_ITA/", num_labels=2)
    model.eval()
    tokenizer_emo = AutoTokenizer.from_pretrained("./local_models/emotion_ITA/")
    model_emo = AutoModelForSequenceClassification.from_pretrained("./local_models/emotion_ITA/", num_labels=4)
    model_emo.eval()


#%%

from transformers import pipeline
import re

generator = pipeline(task="text-classification", model=model, tokenizer=tokenizer, return_all_scores =True)
generator_emo = pipeline(task="text-classification", model=model_emo, tokenizer=tokenizer_emo, return_all_scores =True)

def sentiment_emoji(input_abs):

    if(input_abs ==""):
        return "🤷‍♂️"
        
    res = generator(input_abs)[0]
    res = {res[x]["label"]: res[x]["score"] for x in range(len(res))}
    res["🙂"] = res.pop("positive")
    res["🙁"] = res.pop("negative")
    return res


def emotion_emoji(input_abs):
    if(input_abs ==""):
        return "🤷‍♂️"

    res = generator_emo(input_abs)[0]
    res = {res[x]["label"]: res[x]["score"] for x in range(len(res))}
    res["😃"] = res.pop("joy")
    res["😡"] = res.pop("anger")
    res["😨"] = res.pop("fear")
    res["😟"] = res.pop("sadness")
   
    return res
#%%

import gradio as gr
demo = gr.Blocks()
with demo:
   gr.Markdown("# Analisi sentimento/emozioni del testo italiano")
   with gr.Row():
      with gr.Column():
         text_input = gr.Textbox(placeholder="Scrivi qui")
         button_1 = gr.Button("Invia")
      with gr.Column():
         label_sem = gr.Label()
         label_emo = gr.Label()
      #   gr.Interface(fn=emotion_emoji, inputs=text_input, outputs="label")
   button_1.click(sentiment_emoji, inputs=text_input, outputs=label_sem)
   button_1.click(emotion_emoji, inputs=text_input, outputs=label_emo)
                

demo.launch()