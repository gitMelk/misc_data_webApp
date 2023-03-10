# testing NLP ITA!
A repo in italian for a NLP model testing. Mainly I'm focusing on serving as easily as possible a model using python/js without much back-end.
---

Lo scopo di questa repo è duplice: 
- Testing di framework  python e js per semplici dash di inference
- Testing di framework  python e js come back-end
---
# Gradio: 
Dash interagibile dalla mia repo su [huggingface](https://huggingface.co/spaces/rmelk/gradio_NLP_ITA_dash)



>![gandalf-1](online_res/gandalf_1.png)\
>*Gandalf è arrabbiato.*

<!---
>![pipino-1](online_res/pipino-3.png)\
>*Pipino è felice.*

>![pipino-1](online_res/eowin_1.png)\
>*Eowin ha paura.*

>![pipino-1](online_res/sam_1.png)\
>*Sam è è triste, ma il messagio è positivo.*
-->

---
# MLfLow
I modelli registrati con mlflow permettono di tenere i file binari in locale dopa la prima estrazione da hugging face e possono essere versionati molto semplicemente. Con mlflow è possibile aggiungere delle metriche di esempio e altri elementi per desrivere il modello. 

>![mlflow-1](online_res/mlflow_1.png)\
>*mlflow main interface*






---
Il modello di riferimento per NPL è  [feel-it](https://huggingface.co/MilaNLProc/feel-it-italian-emotion)

<sub><sub>title = "FEEL-IT: Emotion and Sentiment Classification for the Italian Language"\
author = Bianchi, Federico and Nozza, Debora and Hovy, Dirk\
booktitle = "Proceedings of the 11th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis"\
year = 2021\
publisher = "Association for Computational Linguistics"
