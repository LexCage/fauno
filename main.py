from typing import *
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from peft import PeftModel
import torch
import transformers

from huggingface_hub import login


access_token = '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'
login(access_token)

tokenizer = LlamaTokenizer.from_pretrained("baffo32/decapoda-research-llama-7b-hf")
model = LlamaForCausalLM.from_pretrained(
    "baffo32/decapoda-research-llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "buio/Fauno-Italian-LLM-7B")
model.eval()

def evaluate(question):
    prompt = f"La conversazione tra assistente umano e AI. \n[|Human|] {question}.\n[|AI|] "
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            temperature=1,
            top_p=0.95,
            num_beams=4,
            max_context_length_tokens=2048,
        ),
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=300
    )
    output = tokenizer.decode(generation_output.sequences[0]).split("[|AI|]")[1]
    return output

def generate_prompt_with_history(text:str,history: str,tokenizer,max_length=2048):
   # prompt = "The following is a conversation between a human and an AI assistant named Fauno. Fauno is an open-source AI assistant developed by Andrea Bacciu, Giovanni Trappolini, Andrea Santilli and Professor Fabrizio Silvestri members of the RSTLess Research group from Sapienza University of Rome. The human and the AI assistant take turns chatting. Human statements start with [|Umano|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n[|Umano|]Ciao!\n[|AI|]Ciao a te!"
    prompt = 'Sei un generatore di test formati da 5 domande a risposta multipla. Per ogni domanda che scrivi devi associare tre risposte di cui una sola corretta. Indica la risposta corretta. Devi svolgere solo questo compito prendendo le informazioni dal testo che segue:'
    history = ["\n[|Human|]{}\n[|AI|]{}".format(x[0],x[1]) for x in history]
    history.append("\n[|Human|]{}\n[|AI|]".format(text))
    history_text = ""

    for x in history[::-1]:
        if tokenizer(prompt+history_text+x, return_tensors="pt")['input_ids'].size(-1) <= max_length:
            history_text = x + history_text
            flag = True
    if flag:
        return  prompt+history_text,tokenizer(prompt+history_text, return_tensors="pt")
    else:
        return False
    
def is_stop_word_or_prefix(s: str, stop_words: list) -> bool:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return True
        for i in range(1, len(stop_word)):
            if s.endswith(stop_word[:i]):
                return True
    return False

# Greedy Search
def greedy_search(input_ids: torch.Tensor,
                  model: torch.nn.Module,
                  tokenizer: transformers.PreTrainedTokenizer,
                  stop_words: list,
                  max_length: int,
                  temperature: float = 1.0,
                  top_p: float = 1.0,
                  top_k: int = 25) -> Iterator[str]:
    generated_tokens = []
    past_key_values = None
    current_length = 1
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        # apply temperature
        logits /= temperature

        probs = torch.softmax(logits, dim=-1)
        # apply top_p
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0

        # apply top_k
        #if top_k is not None:
        #    probs_sort1, _ = torch.topk(probs_sort, top_k)
        #    min_top_probs_sort = torch.min(probs_sort1, dim=-1, keepdim=True).values
        #    probs_sort = torch.where(probs_sort < min_top_probs_sort, torch.full_like(probs_sort, float(0.0)), probs_sort)

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        input_ids = torch.cat((input_ids, next_token), dim=-1)

        generated_tokens.append(next_token[0].item())
        text = tokenizer.decode(generated_tokens)

        yield text
        if any([x in text for x in stop_words]):
            return
        
@torch.no_grad()
def predict(text:str,
            chatbot,
            history:str="",
            top_p:float=0.95,
            temperature:float=1.0,
            max_length_tokens:int=512,
            max_context_length_tokens:int=2048):
    
    if text=="":
        return ""

    inputs = generate_prompt_with_history(text,history,tokenizer,max_length=max_context_length_tokens)
    prompt,inputs=inputs
    begin_length = len(prompt)

    input_ids = inputs["input_ids"].to(chatbot.device)
    output = []

    for x in greedy_search(input_ids,model,tokenizer,stop_words=["[|Human|]", "[|AI|]"],max_length=max_length_tokens,temperature=temperature,top_p=top_p):
        if is_stop_word_or_prefix(x,["[|Human|]", "[|AI|]"]) is False:
            if "[|Human|]" in x:
                x = x[:x.index("[|Human|]")].strip()
            elif "[| Human |]" in x:
                x = x[:x.index("[| Human |]")].strip()
            elif "[| Umano |]" in x:
                x = x[:x.index("[| Umano |]")].strip()
            elif "[|Umano|]" in x:
                x = x[:x.index("[|Umano|]")].strip()
            if "[|AI|]" in x:
                x = x[:x.index("[|AI|]")].strip()
            x = x.strip(" ")
            output.append(x)
    return output[-1]



text = """Carlo, detto Magno o Carlomagno o Carlo I detto il Grande, dal latino Carolus Magnus, in tedesco Karl der Große, in francese Charlemagne (2 aprile 742 – Aquisgrana, 28 gennaio 814), è stato re dei Franchi dal 768, re dei Longobardi dal 774 e dall'800 primo Imperatore dei Romani, incoronato da papa Leone III nell'antica basilica di San Pietro in Vaticano. L'appellativo Magno gli fu attribuito dal suo biografo Eginardo, che intitolò la sua opera Vita et gesta Caroli Magni.

Figlio di Pipino il Breve e Bertrada di Laon, Carlo divenne re nel 768, alla morte di suo padre. Inizialmente regnò insieme al fratello Carlomanno. La morte improvvisa di quest'ultimo, avvenuta nel 771 in circostanze misteriose, rese Carlo unico sovrano del regno franco. Questo regno fu da lui allargato, grazie a una serie di vittoriose campagne militari, che inclusero la conquista del Regno longobardo, fino a comprendere una vasta parte dell'Europa occidentale.

Il giorno di Natale dell'800, papa Leone III lo incoronò Imperator Augustus, titolo che all'epoca designava l'imperatore dei Romani. Con l'incoronazione a imperatore si ha la fondazione ufficiale dell'Impero carolingio che è considerato secondo alcuni storici la prima fase nella storia del Sacro Romano Impero.[4][5][6] Durante il regno di Carlo Magno si assistette, quindi, alla fine nella storia dell'Europa occidentale del modello giuridico-formale dei Regni romano-germanici in favore di un nuovo modello di Impero. Inoltre, col suo governo diede impulso alla Rinascita carolingia, un periodo di rinascita degli studi politici, teologici e umanistici nell'Europa continentale.

L'impero resistette nella forma datagli da Carlo Magno fin quando fu in vita il figlio Ludovico il Pio. Alla morte di Ludovico, l'impero fu diviso fra i suoi tre eredi: Lotario I, Carlo il Calvo e Ludovico II il Germanico. Tuttavia, la portata delle riforme apportate da Carlo Magno, così come la valenza sacrale della sua persona, influenzarono radicalmente tutta la vita e la politica del continente europeo nei secoli successivi. Per questo motivo, alcuni storici lo definiscono re, padre dell'Europa (Rex Pater Europae).[7]

Tramite il figlio Ludovico il Pio, egli è antenato di tutte le Case Reali Europee, tra cui i Windsor (Re del Regno Unito), i Sassonia-Coburgo-Gotha (Re del Belgio), dei Borboni di Spagna (Re di Spagna), del re di Svezia Carlo XVI Gustavo (in quanto discendente dei Sassonia-Coburgo-Gotha, ma la casa reale di Svezia non deriva dai Carolingi), della Famiglia Granducale del Lussemburgo oltre alle numerose case reali ora non più regnanti, come i Romanov, i Savoia, i Borbone di Francia e varie altre. 

Potresti creare un test di cinque domande sull'argomento "Carlo Magno"? Per ogni domanda, includi tre possibili risposte: una corretta e due errate. Assicurati che le domande coprano vari aspetti della vita, dei successi e dell'importanza storica di Carlo Magno, come la sua incoronazione, le campagne militari, il contributo alla cultura e all'istruzione, e il suo ruolo nell'unificazione dell'Europa. Cerca di variare il tipo di domande tra scelta multipla, vero o falso, e domande specifiche che richiedono una comprensione più profonda dell'argomento.

Assicurati che le risposte errate siano plausibili ma chiaramente distinte dalla risposta corretta, per stimolare l'apprendimento e la riflessione critica. Grazie!"""

print(predict(text, model, temperature=1.0))

