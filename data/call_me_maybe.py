import json
import time
import numpy as np

try:
    from ..llm_sdk import Small_LLM_Model
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from llm_sdk import Small_LLM_Model


FUNCTIONS_DEFS_PATH = "input/functions_definition.json"
TESTS_PATH = "input/function_calling_tests.json"
OUTPUT_PATH = "output/function_calling_results.json"

with open(FUNCTIONS_DEFS_PATH, "r", encoding="utf-8") as f:
    functions_list = json.load(f)
# inicialize tool named null
tools_info = ""

for fn in functions_list:
    tools_info += f"Tool Name: {fn['name']}\n"
    tools_info += f"Description: {fn['description']}\n"
    tools_info += f"Parameters: {fn['parameters']}\n"
    tools_info += "\n"
TOOLS_DESCRIPTION = f"""
You are a tool-calling assistant.

If the user request can be solved with a tool,
respond ONLY with valid JSON.

The JSON should have the following format:
Example:
Every detail:
{tools_info}
fn_unkown: Call this function ONLY if the user request is ambiguous, lacks details, or does not perfectly match the other functions.

Do not add text before or after the JSON.
It must end with the closing curly brace of the JSON.
"""
print(TOOLS_DESCRIPTION)
func_defs = "input/function_calling_tests.json"
func_calls = "input/function_calls.json"
output = "output/function_calls_output.json"


class CallMeMaybe(Small_LLM_Model):
    def __init__(self):
        super().__init__()
        self.id_json_open = self.encode("{")[0].tolist()[0]
        self.id_json_close = self.encode("}")[0].tolist()[0]
        self.id_quote = self.encode('"')[0].tolist()[0]
        self.id_colon = self.encode(":")[0].tolist()[0]

        # load function definitions once (evita opens repetidos)
        with open(FUNCTIONS_DEFS_PATH, "r", encoding="utf-8") as f:
            self.functions = json.load(f)

        # pre-encode tokens fixos para usar input_ids.extend(...) sem inferência
        self.name_token = self.encode('"name": "')[0].tolist()
        self.params_open_token = self.encode(', "parameters": {')[0].tolist()

    def get_func_definitions(self):
        # retorna o cache carregado no __init__
        return self.functions

    def generate(self, input_ids: list, user_request: str) -> str:
        prompt_len = len(input_ids)

        # Escapa corretamente aspas internas no prompt
        safe_user_request = json.dumps(user_request, ensure_ascii=False)
        request = self.encode(f'"prompt": {safe_user_request},')[0].tolist()

        # tokens fixos pre-encodados -> adicionamos com extend
        name = self.name_token
        param = self.params_open_token
        # --- DESCODIFICAÇÃO RESTRITA (CONSTRAINED DECODING) ---
        # Se for o primeiro token da resposta, forçamos a chaveta '{'
        input_ids.extend([self.id_json_open])
        # ------------------------------------------------------
        input_ids.extend(request)
        # Escolhemos o ID com maior probabilidade (agora restrito)
        input_ids.extend(name)
        # Agora queremos forçar a geração do nome da função, que tem de ser um dos nomes das funções disponíveis, por isso vamos fazer um ciclo enquanto o que for gerado for diferente de todas as chaves de func_name, e dentro do ciclo vamos ir restringindo os tokens possíveis para os próximos tokens da função
        logits = self.get_logits_from_input_ids(input_ids)
        constrained_logits = np.full_like(logits, -float("inf"))
        # quero um ciclo enquanto a for diferente de todas as chaves de func_name
        r = ""  # Decodifica os últimos tokens para verificar o que foi gerado
        i = 0
        funcs = self.get_func_definitions()
        func_name = {}
        for f in funcs:
            func_name[f["name"]] = f["description"], f["parameters"], f["returns"]
        # also add null here, same formate as "Tool Name: null\nDescription: A tool that does nothing, used when the user request cannot be solved with any of the other tools.\nParameters: None\n\n"
        func_name["fn_unkown"] = "Function not found", {}, None
        # problem: what if we dont have the function in
        while r not in func_name.keys():
            logits = self.get_logits_from_input_ids(input_ids)
            constrained_logits = np.full_like(logits, -float("inf"))
            for func in func_name.keys():
                if func.startswith(r):
                    func_encoded = self.encode(func)[0].tolist()[i]
                    constrained_logits[func_encoded] = logits[func_encoded]
            logits = constrained_logits
            i += 1
            next_token_id = int(np.argmax(logits))
            r += self.decode([next_token_id])
            input_ids.append(next_token_id)
        i = 0
        a = self.encode('"')[0].tolist()
        input_ids.extend(a)
        while i < len(param):
            logits = self.get_logits_from_input_ids(input_ids)
            constrained_logits = np.full_like(logits, -float("inf"))
            constrained_logits[param[i]] = 10.0  # Prioriza o próximo token do user_request
            logits = constrained_logits
            next_token_id = int(np.argmax(logits))
            input_ids.append(next_token_id)
            i += 1
        # usa as funções carregadas no __init__
        def get_function(name, functions):
            return next((fn for fn in functions if fn["name"] == name), None)

        fn = get_function(r, self.functions)
        params = fn["parameters"] if fn else {}
        # also add null option
        for p, t in params.items():
            key = '"' + p + '": '
            key_encoded = self.encode(key)[0].tolist()
            j = 0
            while j < len(key_encoded):
                logits = self.get_logits_from_input_ids(input_ids)
                constrained_logits = np.full_like(logits, -float("inf"))
                constrained_logits[key_encoded[j]] = 10.0  # Prioriza o próximo token do user_request
                logits = constrained_logits
                next_token_id = int(np.argmax(logits))
                input_ids.append(next_token_id)
                j += 1
            logits = self.get_logits_from_input_ids(input_ids)
            constrained_logits = np.full_like(logits, -float("inf"))
            # Para o valor do parâmetro, quero obrigar a colocar o valor do user_request (que está em request), tem de ser ciclo para compar palarvas e numeros grandes
            r = ""  # Decodifica os últimos tokens para verificar o que foi gerado
            i = 0
            # enquanto r nao estiver em request
            if t["type"] == "string":
                # Para strings, adicionamos aspas no início e no final
                a = self.encode('"')[0].tolist()
                input_ids.extend(a)
            while True:
                logits = self.get_logits_from_input_ids(input_ids)
                next_token_id = int(np.argmax(logits))
                r += self.decode([next_token_id])
                # verificar se, caso for type int, so pode ter numero
                if t["type"] == "number":
                    if not r.isdigit():
                        break
                else:
                    if '"' in r or "{" in r or "}" in r:  # Para strings, verificamos se a aspa de fechamento foi gerada
                        r = r[:-1]  # Remove a aspa de fechamento do valor
                        break
                input_ids.append(next_token_id)
            if t["type"] == "string":
                a = self.encode('"')[0].tolist()
                input_ids.extend(a)
            if p != list(params.keys())[-1]:  # Se não for o último parâmetro, adiciona a vírgula
                a = self.encode(', ')[0].tolist()
                input_ids.extend(a)

        # fecha 'parameters' e o objeto principal
        input_ids.append(self.id_json_close)
        input_ids.append(self.id_json_close)

        # Decodificamos APENAS o que foi gerado (ignorando o prompt inicial)
        generated_text = self.decode(input_ids[prompt_len:])
        return generated_text.strip()

    def call_tool(self):
        start = time.perf_counter()
        results = []

        with open(TESTS_PATH, "r", encoding="utf-8") as f:
            tests = json.load(f)
        tools_ids = self.encode(TOOLS_DESCRIPTION)[0].tolist()
        for call in tests:
            prompt_text = call["prompt"]
            print(f"User request: {prompt_text}")
            prompt_str = "\n\nUser request: " + prompt_text
            prompt_ids = self.encode(prompt_str)[0].tolist()
            input_ids = tools_ids + prompt_ids
            raw = self.generate(input_ids, prompt_text)

            try:
                results.append(json.loads(raw))
                print(f"Generated JSON: {raw}")
            except json.JSONDecodeError as e:
                results.append({
                    "prompt": prompt_text,
                    "error": f"invalid JSON: {e}",
                    "raw": raw
                })

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        elapsed = time.perf_counter() - start
        print(f"Tempo de execução: {elapsed:.4f}s")


# export HF_HOME=/sgoinfre/$(whoami)/hf_cache
call_me_maybe = CallMeMaybe()
call_me_maybe.call_tool()
