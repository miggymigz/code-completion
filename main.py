from tokenizer import Tokenizer

x = Tokenizer()
a = x.tokenize(
    'class TestClass():\n    @classmethod\n    def hello(cls):\n        pass\n')
print(list(a))
