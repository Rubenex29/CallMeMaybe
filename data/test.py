import json
import traceback
import time

# IMPORTANTE: Ajusta o import de acordo com o nome do teu ficheiro principal
# Se o teu código estiver num ficheiro chamado call_me_maybe.py, usa:
from call_me_maybe import CallMeMaybe, TOOLS_DESCRIPTION 

class TestSuite:
    def __init__(self):
        print("⏳ A carregar o modelo para os testes...")
        self.modelo = CallMeMaybe()
        self.tools_ids = self.modelo.encode(TOOLS_DESCRIPTION)[0].tolist()
        self.passados = 0
        self.falhados = 0

    def simular_prompt(self, prompt_text):
        """Função auxiliar para não repetirmos código em cada teste."""
        prompt_str = "\n\nUser request: " + prompt_text
        prompt_ids = self.modelo.encode(prompt_str)[0].tolist()
        input_ids = self.tools_ids + prompt_ids
        
        raw_output = self.modelo.generate(input_ids, prompt_text)
        return json.loads(raw_output)  # Transforma o output num dicionário Python

    # --- INÍCIO DOS TESTES ---

    def test_numeros_negativos(self):
        resultado = self.simular_prompt("What is the sum of -5 and 10?")
        assert resultado["name"] == "fn_add_numbers", "Deveria ter escolhido fn_add_numbers"
        assert resultado["parameters"]["a"] == -5 or resultado["parameters"]["b"] == -5, "Falhou a extração do número negativo."

    def test_numeros_decimais(self):
        resultado = self.simular_prompt("Add 3.14 to 2")
        assert resultado["name"] == "fn_add_numbers", "Deveria ter escolhido fn_add_numbers"
        # Verifica se pelo menos um dos parâmetros tem o valor decimal correto
        params = resultado["parameters"].values()
        assert 3.14 in params, "Falhou a extração do número decimal."

    def test_prompt_ambiguo(self):
        resultado = self.simular_prompt("Hello, can you help me bake a cake?")
        assert resultado["name"] == "fn_unkown", "Deveria ter escolhido fn_unkown para um prompt não relacionado."

    def test_string_vazia(self):
        resultado = self.simular_prompt("")
        assert resultado["name"] == "fn_unkown", "Deveria ter escolhido fn_unkown para uma string vazia."

    def test_caracteres_especiais(self):
        # Testa se a extração de strings (ex: para a função greet ou reverse) lida bem com símbolos
        resultado = self.simular_prompt("Reverse the string '@#%&*!'")
        assert resultado["name"] == "fn_reverse_string", "Deveria ter escolhido fn_reverse_string"
        assert resultado["parameters"]["s"] == "@#%&*!", "Falhou a extração de caracteres especiais."

    # --- MOTOR DE EXECUÇÃO ---

    def correr_todos(self):
        print("\n🚀 A iniciar o Call Me Maybe Test Suite...\n" + "-"*40)
        
        # Lista de todas as funções de teste que criámos acima
        testes = [
            self.test_numeros_negativos,
            self.test_numeros_decimais,
            self.test_prompt_ambiguo,
            self.test_string_vazia,
            self.test_caracteres_especiais
        ]

        start_time = time.perf_counter()

        for teste in testes:
            nome_teste = teste.__name__
            try:
                teste() # Tenta correr a função
                print(f"✅ {nome_teste}: PASSOU")
                self.passados += 1
            except AssertionError as e:
                # O teste falhou num dos nossos 'asserts'
                print(f"❌ {nome_teste}: FALHOU - {e}")
                self.falhados += 1
            except Exception as e:
                # O teste falhou por causa de um erro de Python (ex: o modelo não gerou JSON válido)
                print(f"💥 {nome_teste}: ERRO CRÍTICO - {type(e).__name__}: {e}")
                self.falhados += 1

        elapsed = time.perf_counter() - start_time
        print("-" * 40)
        print(f"🏁 Resumo: {self.passados} Passaram | {self.falhados} Falharam (Tempo: {elapsed:.2f}s)")


if __name__ == "__main__":
    suite = TestSuite()
    suite.correr_todos()